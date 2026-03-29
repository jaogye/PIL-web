"""
Belgium ETL – Step 2: Load age-group target populations from Statbel.

Statbel does not publish age-by-statistical-sector directly. This script:

  1. Downloads the population-by-age-by-MUNICIPALITY file (TF_SOC_POP_STRUCT).
  2. Reads sector → municipality mapping and sector total populations from the DB.
  3. Distributes each municipality's age-group counts proportionally to its
     sectors based on each sector's share of the municipality total population.

Age groups inserted:
    age_0_3    (0–3 years)
    age_6_11   (6–11 years)
    age_12_17  (12–17 years)

Usage:
    python 02_load_target_populations.py \\
        [--db postgresql://lip2:lip2@localhost:5432/lip2_belgium] \\
        [--age-file /path/to/TF_SOC_POP_STRUCT_2024.zip]

Requirements:
    pip install pandas psycopg2-binary requests
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import psycopg2
import requests
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DB = os.getenv("DATABASE_URL_BELGIUM", "postgresql://lip2:lip2@localhost:5432/lip2_belgium")

# Statbel population by age and municipality (tried in order).
# Format: pipe-delimited TXT with columns CD_REFNIS (municipality NIS), CD_AGE, MS_POPULATION.
AGE_MUN_URLS = [
    # Population by place of residence, nationality, marital status, age and sex (municipality level).
    "https://statbel.fgov.be/sites/default/files/files/opendata/bevolking%20naar%20woonplaats%2C%20nationaliteit%20burgelijke%20staat%20%2C%20leeftijd%20en%20geslacht/TF_SOC_POP_STRUCT_2025.zip",
]

AGE_GROUPS = [
    ("age_0_3",   0,   3),
    ("age_6_11",  6,  11),
    ("age_12_17", 12, 17),
]

# Common Statbel column names.
MUN_COL_CANDIDATES = ["CD_REFNIS", "CD_MUNTY_REFNIS", "CD_COMMUNE", "NIS", "REFNIS"]
AGE_COL_CANDIDATES = ["CD_AGE", "CD_LEEFTIJD", "AGE", "LEEFTIJD"]
POP_COL_CANDIDATES = ["MS_POPULATION", "TOTAL", "BEVOLKING", "POPULATION", "MS_POP"]


# ─────────────────────────────────────────────────────────────────────────────
# Download helper
# ─────────────────────────────────────────────────────────────────────────────

def _download(urls: list[str], cache_dir: Path) -> Path | None:
    for url in urls:
        fname = cache_dir / url.split("/")[-1].split("?")[0]
        if fname.exists():
            log.info("Using cached file: %s", fname)
            return fname
        log.info("Downloading: %s", url)
        try:
            r = requests.get(url, timeout=180, stream=True)
            r.raise_for_status()
            with open(fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            log.info("Downloaded to %s", fname)
            return fname
        except Exception as exc:
            log.warning("Failed (%s): %s", url, exc)
    return None


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Load municipality-level age data
# ─────────────────────────────────────────────────────────────────────────────

def _load_municipality_age(path: Path) -> pd.DataFrame:
    """Return DataFrame with columns [municipality_nis, age, population]."""
    log.info("Reading municipality age file: %s", path)

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            txt_names = [n for n in zf.namelist() if n.lower().endswith((".txt", ".csv"))]
            if not txt_names:
                raise ValueError(f"No TXT/CSV inside {path.name}")
            with zf.open(txt_names[0]) as f:
                raw = f.read()
    else:
        raw = path.read_bytes()

    for sep in ("|", ";", ",", "\t"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str, low_memory=False)
            if len(df.columns) > 3:
                log.info("Parsed with separator %r — %d rows, %d cols", sep, len(df), len(df.columns))
                log.info("Columns: %s", list(df.columns))
                break
        except Exception:
            continue
    else:
        raise ValueError("Could not parse municipality age file.")

    mun_col = _find_col(df, MUN_COL_CANDIDATES)
    age_col = _find_col(df, AGE_COL_CANDIDATES)
    pop_col = _find_col(df, POP_COL_CANDIDATES)

    if not all([mun_col, age_col, pop_col]):
        log.error("Columns found: %s", list(df.columns))
        raise ValueError(
            f"Could not detect required columns.\n"
            f"  Municipality: looked for {MUN_COL_CANDIDATES}, found {mun_col}\n"
            f"  Age:          looked for {AGE_COL_CANDIDATES}, found {age_col}\n"
            f"  Population:   looked for {POP_COL_CANDIDATES}, found {pop_col}"
        )

    log.info("Using columns: municipality=%s  age=%s  population=%s", mun_col, age_col, pop_col)

    df = df[[mun_col, age_col, pop_col]].copy()
    df.columns = ["municipality_nis", "age", "population"]
    df["age"]        = pd.to_numeric(df["age"],        errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0.0)
    df["municipality_nis"] = df["municipality_nis"].astype(str).str.strip().str.zfill(5)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Proportional distribution: municipality → sectors
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sector_populations(
    mun_age_df: pd.DataFrame,
    conn,
) -> dict[int, dict[str, float]]:
    """
    For each sector, compute age-group populations proportionally from its
    municipality's age distribution and the sector's share of the municipality total.

    Returns {census_area_id: {tp_code: population}}.
    """
    # Load sectors with their total population (all_ages) and municipality NIS.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                ca.id,
                ca.parish_code           AS municipality_nis,
                cap.population           AS total_pop
            FROM census_areas ca
            JOIN census_areas_population cap
                ON cap.census_area_id = ca.id
            JOIN target_population tp
                ON tp.id = cap.target_population_id
                AND tp.code = 'all_ages'
        """)
        sectors = cur.fetchall()  # [(area_id, municipality_nis, total_pop)]

    if not sectors:
        raise RuntimeError(
            "No all_ages population found in census_areas_population. "
            "Run 01_load_census_areas.py first to populate total population."
        )

    log.info("Loaded %d sectors from DB", len(sectors))

    # Municipality total population from the same data (sum of all ages).
    mun_totals = (
        mun_age_df.groupby("municipality_nis")["population"].sum().to_dict()
    )

    # Age-group sums per municipality.
    mun_age_groups: dict[str, dict[str, float]] = {}
    for tp_code, min_age, max_age in AGE_GROUPS:
        mask = (mun_age_df["age"] >= min_age) & (mun_age_df["age"] <= max_age)
        agged = mun_age_df[mask].groupby("municipality_nis")["population"].sum()
        for nis, pop in agged.items():
            mun_age_groups.setdefault(nis, {})[tp_code] = float(pop)

    # Distribute proportionally to sectors.
    result: dict[int, dict[str, float]] = {}
    unmatched = 0

    for area_id, nis_raw, sector_total in sectors:
        # parish_code is stored as the 5-digit municipality NIS in Belgium.
        nis = str(nis_raw).strip().zfill(5)
        mun_total = mun_totals.get(nis, 0.0)

        if mun_total == 0.0 or sector_total is None or sector_total == 0.0:
            unmatched += 1
            continue

        ratio = sector_total / mun_total
        groups = mun_age_groups.get(nis, {})

        for tp_code, min_age, max_age in AGE_GROUPS:
            mun_group_pop = groups.get(tp_code, 0.0)
            result.setdefault(area_id, {})[tp_code] = round(mun_group_pop * ratio, 2)

    if unmatched:
        log.warning("%d sectors had no municipality match or zero population (skipped)", unmatched)

    log.info("Computed age-group populations for %d sectors", len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Database upsert
# ─────────────────────────────────────────────────────────────────────────────

def _upsert(conn, sector_pops: dict[int, dict[str, float]]) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT code, id FROM target_population")
        tp_ids = {row[0]: row[1] for row in cur.fetchall()}

    rows: list[tuple[int, int, float]] = []
    for area_id, groups in sector_pops.items():
        for tp_code, pop in groups.items():
            tp_id = tp_ids.get(tp_code)
            if tp_id is None:
                continue
            rows.append((area_id, tp_id, pop))

    if not rows:
        log.error("No rows to insert.")
        return

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO census_areas_population (census_area_id, target_population_id, population)
            VALUES %s
            ON CONFLICT (census_area_id, target_population_id)
            DO UPDATE SET population = EXCLUDED.population
            """,
            rows,
        )
    conn.commit()
    log.info("Upserted %d census_areas_population rows", len(rows))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db",       default=DEFAULT_DB, help="PostgreSQL connection string")
    parser.add_argument("--age-file", default=None,
                        help="Pre-downloaded TF_SOC_POP_STRUCT ZIP or TXT file")
    parser.add_argument(
        "--cache-dir",
        default=str(Path(tempfile.gettempdir()) / "lip2_belgium"),
        help="Directory to cache downloaded files",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.age_file:
        age_path = Path(args.age_file)
    else:
        age_path = _download(AGE_MUN_URLS, cache_dir)
        if not age_path:
            log.error(
                "Could not download municipality age data. Download manually from:\n"
                "  https://statbel.fgov.be/en/open-data/population-sex-age-group-marital-status-and-municipality\n"
                "and pass --age-file <path>"
            )
            sys.exit(1)

    mun_age_df = _load_municipality_age(age_path)
    log.info("Municipality age data: %d rows", len(mun_age_df))

    db_url = args.db.replace("postgresql+asyncpg://", "postgresql://")
    log.info("Connecting to database: %s", db_url)
    conn = psycopg2.connect(db_url)
    try:
        sector_pops = _compute_sector_populations(mun_age_df, conn)
        _upsert(conn, sector_pops)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    log.info("Done.")


if __name__ == "__main__":
    main()
