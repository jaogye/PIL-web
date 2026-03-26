"""
Belgium ETL – Step 2: Load facilities (schools and hospitals).

Source: Geofabrik Belgium free shapefile (OSM data).
  https://download.geofabrik.de/europe/belgium-latest-free.shp.zip
  Layer: gis_osm_pois_free_1.shp
  Relevant fclass values:
    school, college, university       → 'school' / 'high_school'
    hospital, clinic, doctors, pharmacy → 'hospital' / 'health_center'

Assigns each facility to the nearest census area centroid (ST_Distance).

Usage:
    python 02_load_facilities.py [--db postgresql://lip2:lip2@localhost:5432/lip2_belgium]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

GEOFABRIK_URL = "https://download.geofabrik.de/europe/belgium-latest-free.shp.zip"
DEFAULT_DB = os.getenv("DATABASE_URL_BELGIUM",
                       "postgresql://lip2:lip2@localhost:5432/lip2_belgium")

# OSM fclass → facility_type mapping
FCLASS_MAP = {
    "school":         "school",
    "kindergarten":   "school",
    "college":        "high_school",
    "university":     "high_school",
    "hospital":       "hospital",
    "clinic":         "health_center",
    "doctors":        "health_center",
    "dentist":        "health_center",
    "pharmacy":       "health_center",
}


def _download_geofabrik(cache_dir: Path) -> Path:
    dest = cache_dir / "belgium-latest-free.shp.zip"

    def _size(p: Path) -> int:
        return p.stat().st_size if p.exists() else 0

    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        existing = _size(dest)
        headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
        if existing > 0:
            log.info("Resuming download at %d MB (attempt %d) …", existing // 1048576, attempt + 1)
        else:
            log.info("Downloading Geofabrik Belgium SHP (attempt %d) …", attempt + 1)
        try:
            r = requests.get(GEOFABRIK_URL, headers=headers, timeout=120, stream=True)
            if r.status_code == 416:
                # Range not satisfiable – file already complete or server ignores ranges
                if existing > 1_000_000:
                    log.info("File appears complete (%d MB).", existing // 1048576)
                    return dest
                dest.unlink(missing_ok=True)
                continue
            r.raise_for_status()
            total_header = r.headers.get("content-length")
            mode = "ab" if existing > 0 and r.status_code == 206 else "wb"
            if mode == "wb" and existing > 0:
                dest.unlink()
                existing = 0
            downloaded = existing
            with open(dest, mode) as f:
                for chunk in r.iter_content(chunk_size=2 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_header:
                        total = int(total_header) + existing
                        print(f"\r  {downloaded * 100 // max(total,1)}%  ({downloaded // 1048576} MB)", end="", flush=True)
                    else:
                        print(f"\r  {downloaded // 1048576} MB downloaded", end="", flush=True)
            print()
            log.info("Download complete: %d MB", _size(dest) // 1048576)
            return dest
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as exc:
            log.warning("  Download interrupted (%s). Will retry …", exc)
            import time; time.sleep(3)

    raise RuntimeError(f"Failed to download {GEOFABRIK_URL} after {MAX_RETRIES} attempts")


def load_pois(zip_path: Path, extract_dir: Path) -> gpd.GeoDataFrame:
    """Extract and return the POI layer from the Geofabrik SHP ZIP."""
    poi_shp = extract_dir / "gis_osm_pois_free_1.shp"
    if not poi_shp.exists():
        log.info("Extracting POI shapefile …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if "gis_osm_pois_free_1" in name:
                    zf.extract(name, extract_dir)
    gdf = gpd.read_file(poi_shp)
    log.info("POI layer loaded: %d features", len(gdf))
    return gdf


def filter_facilities(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    mask = gdf["fclass"].isin(FCLASS_MAP)
    fac = gdf[mask].copy()
    fac["facility_type"] = fac["fclass"].map(FCLASS_MAP)
    if fac.crs and fac.crs.to_epsg() != 4326:
        fac = fac.to_crs(epsg=4326)
    log.info("Facilities after filter: %d", len(fac))
    for ft in fac["facility_type"].unique():
        log.info("  %-20s  %d", ft, (fac["facility_type"] == ft).sum())
    return fac


def load_census_centroids(conn) -> tuple[np.ndarray, list[int]]:
    """Return (Nx2 lon/lat array, list of census_area ids)."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, x, y FROM census_areas WHERE x IS NOT NULL AND y IS NOT NULL")
        rows = cur.fetchall()
    ids = [r[0] for r in rows]
    xy  = np.array([[r[1], r[2]] for r in rows], dtype=np.float64)
    return xy, ids


def assign_to_nearest_census(fac: gpd.GeoDataFrame,
                              census_xy: np.ndarray,
                              census_ids: list[int]) -> list[int | None]:
    """
    For each facility, find the nearest census centroid by Euclidean distance.
    Returns list of census_area_id (same length as fac).
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(census_xy)
    coords = np.column_stack([fac.geometry.x, fac.geometry.y])
    _, idxs = tree.query(coords, k=1)
    return [census_ids[i] for i in idxs]


def insert_facilities(cur, fac: gpd.GeoDataFrame, census_area_ids: list[int | None]) -> None:
    rows = []
    for (_, row), cid in zip(fac.iterrows(), census_area_ids):
        lon = row.geometry.x
        lat = row.geometry.y
        name = str(row.get("name", "")).strip() or None
        ftype = row["facility_type"]
        rows.append((name, ftype, "existing", 0.0, cid,
                     f"POINT({lon} {lat})"))

    log.info("Inserting %d facilities …", len(rows))
    BATCH = 2000
    for i in range(0, len(rows), BATCH):
        execute_values(cur, """
            INSERT INTO facilities (name, facility_type, status, capacity, census_area_id, geom)
            VALUES %s
        """, [
            (r[0], r[1], r[2], r[3], r[4], r[5])   # r[5] is WKT string
            for r in rows[i:i+BATCH]
        ], template="(%s,%s,%s,%s,%s,ST_GeomFromText(%s,4326))")
    log.info("  Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--cache-dir",
                        default=str(Path(tempfile.gettempdir()) / "lip2_belgium"))
    parser.add_argument("--shp-zip", help="Path to pre-downloaded Geofabrik SHP ZIP")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download
    zip_path = Path(args.shp_zip) if args.shp_zip else _download_geofabrik(cache_dir)

    # Load and filter
    extract_dir = cache_dir / "geofabrik"
    extract_dir.mkdir(exist_ok=True)
    pois = load_pois(zip_path, extract_dir)
    fac  = filter_facilities(pois)

    if len(fac) == 0:
        log.error("No facilities found after filter. Check fclass values.")
        sys.exit(1)

    # DB
    db_url = args.db.replace("postgresql+asyncpg://", "postgresql://")
    conn = psycopg2.connect(db_url)
    try:
        census_xy, census_ids = load_census_centroids(conn)
        log.info("Census centroids loaded: %d", len(census_ids))

        census_area_ids = assign_to_nearest_census(fac, census_xy, census_ids)

        with conn.cursor() as cur:
            insert_facilities(cur, fac, census_area_ids)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM facilities")
            log.info("Total facilities in DB: %d", cur.fetchone()[0])
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
