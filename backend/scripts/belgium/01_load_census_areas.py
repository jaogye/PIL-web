"""
Belgium ETL – Step 1: Load political divisions and census areas.

Downloads from Statbel:
  • Statistical sectors 2024 (GeoJSON, Lambert 2008 → WGS84)
  • Population by statistical sector 2024 (TXT/ZIP)

Loads into lip2_belgium:
  • political_division  (region > province > municipality)
  • census_areas        (one row per statistical sector)

Usage:
    python 01_load_census_areas.py [--db postgresql://lip2:lip2@localhost:5432/lip2_belgium]
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import psycopg2
import requests
from psycopg2.extras import execute_values
from shapely.geometry import mapping

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Download URLs ─────────────────────────────────────────────────────────────
# Statbel statistical sectors 2024 (GeoJSON, Lambert 2008)
# If any URL fails, the script tries the next one.
SECTOR_URLS = [
    # GeoJSON, Lambert 2008 (EPSG:3812) – will be reprojected to WGS84
    "https://statbel.fgov.be/sites/default/files/files/opendata/Statistische%20sectoren/"
    "sh_statbel_statistical_sectors_3812_20240101.geojson.zip",
    # Fallback: Lambert 1972 (EPSG:31370)
    "https://statbel.fgov.be/sites/default/files/files/opendata/Statistische%20sectoren/"
    "sh_statbel_statistical_sectors_31370_20240101.geojson.zip",
]

# Statbel population by statistical sector 2024
POP_URLS = [
    "https://statbel.fgov.be/sites/default/files/files/opendata/bevolking/sectoren/"
    "OPENDATA_SECTOREN_2024.zip",
]

# ── Default DB ────────────────────────────────────────────────────────────────
DEFAULT_DB = os.getenv("DATABASE_URL_BELGIUM",
                       "postgresql://lip2:lip2@localhost:5432/lip2_belgium")


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download(urls: list[str], label: str, cache_dir: Path) -> Path | None:
    """Try each URL in order; return path to downloaded file or None."""
    for url in urls:
        fname = cache_dir / url.split("/")[-1].split("?")[0]
        if fname.exists():
            log.info("  %s: using cached %s", label, fname.name)
            return fname
        log.info("  %s: downloading %s …", label, url)
        try:
            r = requests.get(url, timeout=120, stream=True)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r    {pct}% ({downloaded // 1048576} MB)", end="", flush=True)
            print()
            log.info("  %s: saved to %s", label, fname)
            return fname
        except Exception as exc:
            log.warning("  %s: failed (%s)", label, exc)
            if fname.exists():
                fname.unlink()
    return None


def _extract_geojson_or_shp(zip_path: Path, extract_dir: Path) -> Path:
    """Extract the first GeoJSON or shapefile from a ZIP archive (handles subfolders)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # Prefer GeoJSON
        for name in names:
            if name.lower().endswith(".geojson"):
                zf.extract(name, extract_dir)
                return extract_dir / name
        # Fall back to shapefile (need all components)
        shps = [n for n in names if n.lower().endswith(".shp")]
        if shps:
            for name in names:
                try:
                    zf.extract(name, extract_dir)
                except Exception:
                    pass
            return extract_dir / shps[0]
        raise ValueError(f"No GeoJSON or .shp found in {zip_path.name}. Contents: {names[:10]}")


def _find_pop_csv(zip_path: Path, extract_dir: Path) -> Path:
    """Extract the first TXT or CSV population file from the ZIP."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for name in names:
            if name.lower().endswith((".txt", ".csv")):
                zf.extract(name, extract_dir)
                return extract_dir / name
        raise ValueError(f"No TXT/CSV in {zip_path.name}. Contents: {names}")


# ─────────────────────────────────────────────────────────────────────────────
# Sector geometry loading
# ─────────────────────────────────────────────────────────────────────────────

# Known column names used by Statbel (vary slightly between years).
_SECTOR_COL_CANDIDATES  = ["CD_SECTOR", "cs_sector", "CD_SECTOR_REFNIS"]
_MUNTY_COL_CANDIDATES   = ["CD_MUNTY_REFNIS", "cs_municip", "CD_MUNTY"]
_PROV_COL_CANDIDATES    = ["CD_PROV_REFNIS", "cs_province", "CD_PROV"]
_RGN_COL_CANDIDATES     = ["CD_RGN_REFNIS", "cs_region", "CD_RGN"]
_SECTOR_NL_CANDIDATES   = ["TX_SECTOR_DESCR_NL", "tx_sector_nl", "TX_SECTOR_NL"]
_MUNTY_NL_CANDIDATES    = ["TX_MUNTY_DESCR_NL", "tx_munty_nl", "TX_MUNTY_NL"]
_PROV_NL_CANDIDATES     = ["TX_PROV_DESCR_NL",  "tx_prov_nl",  "TX_PROV_NL"]
_RGN_NL_CANDIDATES      = ["TX_RGN_DESCR_NL",   "tx_rgn_nl",   "TX_RGN_NL"]
_MUNTY_FR_CANDIDATES    = ["TX_MUNTY_DESCR_FR",  "tx_munty_fr", "TX_MUNTY_FR"]
_PROV_FR_CANDIDATES     = ["TX_PROV_DESCR_FR",   "tx_prov_fr",  "TX_PROV_FR"]
_RGN_FR_CANDIDATES      = ["TX_RGN_DESCR_FR",    "tx_rgn_fr",   "TX_RGN_FR"]


def _pick(df: pd.DataFrame, candidates: list[str], default: str | None = None) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return default


def load_sectors(path: Path) -> gpd.GeoDataFrame:
    """Load and reproject the statistical sectors file."""
    log.info("Reading sector file: %s", path.name)
    gdf = gpd.read_file(path)
    log.info("  CRS: %s   rows: %d   columns: %s", gdf.crs, len(gdf), list(gdf.columns[:10]))

    # Reproject to WGS84 if needed
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        log.info("  Reprojecting %s → EPSG:4326 …", gdf.crs.to_epsg())
        gdf = gdf.to_crs(epsg=4326)

    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# Population CSV loading
# ─────────────────────────────────────────────────────────────────────────────

def load_population(path: Path) -> dict[str, float]:
    """Return {sector_code: total_population} from Statbel population file."""
    log.info("Reading population file: %s", path.name)
    # Try pipe or semicolon separator; Statbel uses |
    for sep in ["|", ";", ","]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", dtype=str, low_memory=False)
            if len(df.columns) > 2:
                log.info("  Detected separator: %r   columns: %s", sep, list(df.columns[:8]))
                break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not parse population file: {path}")

    # Find sector code column — CD_SECTOR first (9-char statistical sector code)
    sector_col = None
    for c in ["CD_SECTOR", "cs_sector", "CD_REFNIS", "SECTOR"]:
        if c in df.columns or c.lower() in {x.lower() for x in df.columns}:
            # Case-insensitive pick
            for col in df.columns:
                if col.lower() == c.lower():
                    sector_col = col
                    break
            if sector_col:
                break

    if not sector_col:
        # Last resort: first column
        sector_col = df.columns[0]
        log.warning("  Sector code column not found, using first column: %s", sector_col)

    # Find total population column
    total_col = None
    for c in ["TOTAL", "MS_POPULATION", "BEVOLKING", "POPULATION", "POP_TOTAL"]:
        for col in df.columns:
            if col.upper() == c.upper():
                total_col = col
                break
        if total_col:
            break

    if not total_col:
        # Try: last numeric-looking column
        numeric_cols = [c for c in df.columns if c not in (sector_col,)]
        # Pick column with name containing "tot" or "pop"
        for c in numeric_cols:
            if any(kw in c.lower() for kw in ["tot", "pop", "bev"]):
                total_col = c
                break
        if not total_col:
            total_col = numeric_cols[-1] if numeric_cols else None

    if not total_col:
        raise ValueError(f"Could not find population column in {path}. Columns: {list(df.columns)}")

    log.info("  sector_col=%s  total_col=%s  rows=%d", sector_col, total_col, len(df))
    df["_pop"] = pd.to_numeric(df[total_col], errors="coerce").fillna(0)
    return dict(zip(df[sector_col].str.strip(), df["_pop"]))


# ─────────────────────────────────────────────────────────────────────────────
# DB insert helpers
# ─────────────────────────────────────────────────────────────────────────────

def insert_political_divisions(cur, gdf: gpd.GeoDataFrame) -> dict[str, int]:
    """
    Insert region > province > municipality hierarchy.
    Returns {municipality_nis_code: political_division.id}.
    """
    c_sector  = _pick(gdf, _SECTOR_COL_CANDIDATES)
    c_munty   = _pick(gdf, _MUNTY_COL_CANDIDATES)
    c_prov    = _pick(gdf, _PROV_COL_CANDIDATES)
    c_rgn     = _pick(gdf, _RGN_COL_CANDIDATES)
    c_munty_nl = _pick(gdf, _MUNTY_NL_CANDIDATES)
    c_munty_fr = _pick(gdf, _MUNTY_FR_CANDIDATES)
    c_prov_nl  = _pick(gdf, _PROV_NL_CANDIDATES)
    c_prov_fr  = _pick(gdf, _PROV_FR_CANDIDATES)
    c_rgn_nl   = _pick(gdf, _RGN_NL_CANDIDATES)
    c_rgn_fr   = _pick(gdf, _RGN_FR_CANDIDATES)

    def _name(row, nl_col, fr_col):
        nl = str(row[nl_col]).strip() if nl_col and nl_col in row and pd.notna(row[nl_col]) else ""
        fr = str(row[fr_col]).strip() if fr_col and fr_col in row and pd.notna(row[fr_col]) else ""
        if nl and fr and nl != fr:
            return f"{nl} / {fr}"
        return nl or fr or "?"

    log.info("Collecting unique regions / provinces / municipalities …")

    # Build sets of unique entries
    regions   = {}   # code → name
    provinces = {}   # code → (name, region_code)
    munties   = {}   # nis5 → (name, prov_code)

    for _, row in gdf.iterrows():
        r_code = str(row[c_rgn]).strip()  if c_rgn  else "??"
        p_code = str(row[c_prov]).strip() if c_prov else "??"
        m_code = str(row[c_munty]).strip() if c_munty else "?????"
        r_name = _name(row, c_rgn_nl, c_rgn_fr)
        p_name = _name(row, c_prov_nl, c_prov_fr)
        m_name = _name(row, c_munty_nl, c_munty_fr)
        if r_code not in regions:
            regions[r_code] = r_name
        if p_code not in provinces:
            provinces[p_code] = (p_name, r_code)
        if m_code not in munties:
            munties[m_code] = (m_name, p_code)

    log.info("  regions=%d  provinces=%d  municipalities=%d",
             len(regions), len(provinces), len(munties))

    # Insert regions
    rgn_id: dict[str, int] = {}
    for code, name in sorted(regions.items()):
        cur.execute(
            "INSERT INTO political_division (code, name, level) "
            "VALUES (%s, %s, 'region') ON CONFLICT (code) DO UPDATE SET name=EXCLUDED.name "
            "RETURNING id",
            (f"R_{code}", name),
        )
        rgn_id[code] = cur.fetchone()[0]

    # Insert provinces
    prov_id: dict[str, int] = {}
    for code, (name, r_code) in sorted(provinces.items()):
        cur.execute(
            "INSERT INTO political_division (code, name, level, parent_id) "
            "VALUES (%s, %s, 'province', %s) ON CONFLICT (code) DO UPDATE SET name=EXCLUDED.name "
            "RETURNING id",
            (f"P_{code}", name, rgn_id.get(r_code)),
        )
        prov_id[code] = cur.fetchone()[0]

    # Insert municipalities
    mun_id: dict[str, int] = {}
    for code, (name, p_code) in sorted(munties.items()):
        cur.execute(
            "INSERT INTO political_division (code, name, level, parent_id) "
            "VALUES (%s, %s, 'municipality', %s) ON CONFLICT (code) DO UPDATE SET name=EXCLUDED.name "
            "RETURNING id",
            (code, name, prov_id.get(p_code)),
        )
        mun_id[code] = cur.fetchone()[0]

    return mun_id


def insert_census_areas(cur, gdf: gpd.GeoDataFrame,
                        pop_map: dict[str, float],
                        mun_id: dict[str, int]) -> None:
    """Insert census areas (one row per statistical sector)."""
    c_sector  = _pick(gdf, _SECTOR_COL_CANDIDATES)
    c_munty   = _pick(gdf, _MUNTY_COL_CANDIDATES)
    c_prov    = _pick(gdf, _PROV_COL_CANDIDATES)
    c_sector_nl = _pick(gdf, _SECTOR_NL_CANDIDATES)

    log.info("Inserting %d census areas …", len(gdf))
    rows = []
    missing_pop = 0

    for _, row in gdf.iterrows():
        sector_code = str(row[c_sector]).strip() if c_sector else "?"
        munty_code  = str(row[c_munty]).strip()  if c_munty  else "?????"
        prov_code   = str(row[c_prov]).strip()   if c_prov   else "??"
        name        = str(row[c_sector_nl]).strip() if c_sector_nl else None

        # Population from population CSV (join on sector code)
        pop = pop_map.get(sector_code)
        if pop is None:
            # Some files use leading zeros differently
            pop = pop_map.get(sector_code.lstrip("0")) or 0.0
            missing_pop += 1

        # Compute centroid from polygon geometry
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        centroid = geom.centroid
        lon, lat = centroid.x, centroid.y
        wkt = f"POINT({lon} {lat})"

        # province_code: NIS province = first 2 chars of municipality NIS
        province_code = munty_code[:2] if len(munty_code) >= 2 else prov_code
        # canton_code: NIS arrondissement ≈ first 4 chars of municipality code (no standard)
        #   We store the municipality NIS as canton_code (best available)
        canton_code  = munty_code[:4] if len(munty_code) >= 4 else munty_code
        parish_code  = munty_code

        rows.append((
            sector_code,
            name,
            province_code,
            canton_code,
            parish_code,
            float(pop),
            0.0,           # capacity
            wkt,
            lon,
            lat,
            mun_id.get(munty_code),
        ))

    if missing_pop:
        log.warning("  %d sectors had no population data (population=0)", missing_pop)

    BATCH = 2000
    for i in range(0, len(rows), BATCH):
        execute_values(cur, """
            INSERT INTO census_areas
                (area_code, name, province_code, canton_code, parish_code,
                 capacity,
                 geom, x, y, parish_id)
            VALUES %s
            ON CONFLICT (area_code) DO UPDATE SET
                parish_id = EXCLUDED.parish_id
        """, [
            (r[0], r[1], r[2], r[3], r[4],
             r[6],
             r[7], r[8], r[9], r[10])      # r[7] is WKT string; r[5]=total pop handled below
            for r in rows[i:i+BATCH]
        ], template="(%s,%s,%s,%s,%s,%s,ST_GeomFromText(%s,4326),%s,%s,%s)")
    log.info("  Inserted %d census areas", len(rows))

    # Insert total population into census_areas_population (all_ages group).
    cur.execute("SELECT id FROM target_population WHERE code = 'all_ages'")
    tp_row = cur.fetchone()
    if tp_row:
        tp_id = tp_row[0]
        area_codes = [r[0] for r in rows]
        cur.execute(
            "SELECT area_code, id FROM census_areas WHERE area_code = ANY(%s)",
            (area_codes,),
        )
        area_id_map = {row[0]: row[1] for row in cur.fetchall()}
        pop_rows = [
            (area_id_map[r[0]], tp_id, r[5])
            for r in rows
            if r[0] in area_id_map
        ]
        if pop_rows:
            execute_values(
                cur,
                """
                INSERT INTO census_areas_population (census_area_id, target_population_id, population)
                VALUES %s
                ON CONFLICT (census_area_id, target_population_id)
                DO UPDATE SET population = EXCLUDED.population
                """,
                pop_rows,
            )
            log.info("  Inserted %d all_ages population rows", len(pop_rows))
    else:
        log.warning("  target_population 'all_ages' not found — skipping census_areas_population insert")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB, help="PostgreSQL connection URL")
    parser.add_argument("--cache-dir", default=str(Path(tempfile.gettempdir()) / "lip2_belgium"),
                        help="Directory to cache downloaded files")
    parser.add_argument("--sector-file", help="Path to pre-downloaded sector ZIP or GeoJSON (optional)")
    parser.add_argument("--pop-file",    help="Path to pre-downloaded population ZIP or TXT (optional)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("Cache directory: %s", cache_dir)

    # ── 1. Download or locate sector file ──
    if args.sector_file:
        sector_zip = Path(args.sector_file)
    else:
        sector_zip = _download(SECTOR_URLS, "sectors", cache_dir)
        if not sector_zip:
            log.error("Could not download sector file. Please download manually from "
                      "https://statbel.fgov.be/en/open-data/statistical-sectors-2024 "
                      "and pass --sector-file <path>")
            sys.exit(1)

    # ── 2. Extract and load sectors ──
    extract_dir = cache_dir / "sectors2"
    extract_dir.mkdir(exist_ok=True)
    if sector_zip.suffix.lower() == ".zip":
        geo_path = _extract_geojson_or_shp(sector_zip, extract_dir)
    else:
        geo_path = sector_zip

    gdf = load_sectors(geo_path)

    # ── 3. Download or locate population file ──
    if args.pop_file:
        pop_zip = Path(args.pop_file)
    else:
        pop_zip = _download(POP_URLS, "population", cache_dir)
        if not pop_zip:
            log.warning("Population file not downloaded. demand will be 0 for all areas.")
            pop_zip = None

    pop_map: dict[str, float] = {}
    if pop_zip:
        extract_pop_dir = cache_dir / "population"
        extract_pop_dir.mkdir(exist_ok=True)
        if pop_zip.suffix.lower() == ".zip":
            pop_csv = _find_pop_csv(pop_zip, extract_pop_dir)
        else:
            pop_csv = pop_zip
        pop_map = load_population(pop_csv)
        log.info("Population loaded: %d sector entries", len(pop_map))

    # ── 4. DB operations ──
    db_url = args.db.replace("postgresql+asyncpg://", "postgresql://")
    log.info("Connecting to %s …", db_url)
    conn = psycopg2.connect(db_url)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            log.info("Inserting political divisions …")
            mun_id = insert_political_divisions(cur, gdf)
            conn.commit()
            log.info("Political divisions committed.")

            log.info("Inserting census areas …")
            insert_census_areas(cur, gdf, pop_map, mun_id)
            conn.commit()
            log.info("Census areas committed.")

        # Verify
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM political_division")
            n_div = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM census_areas")
            n_ca = cur.fetchone()[0]
            cur.execute(
                "SELECT COALESCE(SUM(population), 0) FROM census_areas_population cap "
                "JOIN target_population tp ON tp.id = cap.target_population_id "
                "WHERE tp.code = 'all_ages'"
            )
            total_dem = cur.fetchone()[0]
        log.info("Done.  political_division=%d  census_areas=%d  total_population=%.0f",
                 n_div, n_ca, total_dem or 0)

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
