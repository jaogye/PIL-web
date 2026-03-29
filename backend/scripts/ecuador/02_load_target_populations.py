"""
Ecuador ETL – Step 2: Load age-group target populations from CPV 2010.

Reads CPV2010S_CSV_Poblacion_1.csv (Censo de Población y Vivienda 2010),
aggregates population by parroquia and age group, then distributes each
parroquia's totals proportionally to the sector-level census areas.

CPV CSV columns used:
    I01  – Provincia code (2 digits)
    I02  – Cantón code   (2 digits)
    I03  – Parroquia code (2 digits)
    P03  – Age in years  (integer, 3-digit zero-padded)

Parroquia code built as: I01 + I02 + I03  →  e.g. "010150" (6 digits)

Census areas in the DB use 12-digit sector codes (province+canton+parish+
zone+sector).  The first 6 characters of area_code equal the parroquia code,
so each parroquia's CPV population is distributed to its constituent sectors
weighted by the sector's existing all_ages population (already loaded by
migration 004 from the original census shapefile demand column).

Sectors with no all_ages data receive an equal share of their parroquia total.

Age groups inserted:
    age_0_3    (0–3 years)
    age_6_11   (6–11 years)
    age_12_17  (12–17 years)
    all_ages   (total population — overwrites migration-004 value with CPV count)

Usage:
    python 02_load_target_populations.py \\
        [--csv  "C:/Users/.../CPV2010S_CSV_Nacional/CPV2010S_CSV_Poblacion_1.csv"] \\
        [--db   postgresql://lip2:lip2@localhost:5432/lip2_ecuador]

Requirements:
    pip install pandas psycopg2-binary
"""

from __future__ import annotations

import argparse
import logging
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

try:
    import zipfile_deflate64  # noqa: F401  (DEFLATE64 support for Windows ZIPs)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DB  = os.getenv("DATABASE_URL_ECUADOR", "postgresql://lip2:lip2@localhost:5432/lip2_ecuador")
DEFAULT_CSV = os.getenv(
    "CPV_CSV_PATH",
    r"C:\Users\jaoxx\Documents\datasets\CPV2010S_CSV_Nacional\CPV2010S_CSV_Poblacion_1.csv",
)

# Age groups matching target_population codes in the DB.
AGE_GROUPS = [
    ("age_0_3",   0,   3),
    ("age_6_11",  6,  11),
    ("age_12_17", 12, 17),
]

CHUNK_SIZE = 500_000


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: aggregate CPV at parroquia level
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_by_parroquia(path: str) -> dict[str, dict[str, float]]:
    """
    Stream through the CSV (or ZIP-wrapped CSV) in chunks and accumulate
    population counts keyed by 6-digit parroquia code.

    Returns {parroquia_code: {tp_code: count}}.
    """
    p = Path(path)
    if p.suffix.lower() == ".zip":
        log.info("Opening ZIP: %s", path)
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV file found inside {p.name}")
            inner = csv_names[0]
            log.info("Reading CSV inside ZIP: %s", inner)
            raw_file = zf.open(inner)
    else:
        log.info("Reading CSV directly: %s", path)
        raw_file = open(path, "rb")

    totals:      dict[str, float]              = {}
    age_buckets: dict[str, dict[str, float]]   = {tp: {} for tp, _, _ in AGE_GROUPS}

    with raw_file:
        reader = pd.read_csv(
            raw_file,
            usecols=["I01", "I02", "I03", "P03"],
            dtype=str,
            chunksize=CHUNK_SIZE,
        )

        for chunk_num, chunk in enumerate(reader, start=1):
            chunk["parroquia_code"] = (
                chunk["I01"].str.zfill(2)
                + chunk["I02"].str.zfill(2)
                + chunk["I03"].str.zfill(2)
            )
            chunk["age"] = pd.to_numeric(chunk["P03"], errors="coerce")

            for code, cnt in chunk.groupby("parroquia_code").size().items():
                totals[code] = totals.get(code, 0.0) + float(cnt)

            for tp_code, min_age, max_age in AGE_GROUPS:
                mask = (chunk["age"] >= min_age) & (chunk["age"] <= max_age)
                for code, cnt in chunk[mask].groupby("parroquia_code").size().items():
                    b = age_buckets[tp_code]
                    b[code] = b.get(code, 0.0) + float(cnt)

            log.info("  Chunk %d processed (%d M rows so far)",
                     chunk_num, chunk_num * CHUNK_SIZE // 1_000_000)

    result: dict[str, dict[str, float]] = {}
    for code, pop in totals.items():
        result.setdefault(code, {})["all_ages"] = pop
    for tp_code, bucket in age_buckets.items():
        for code, pop in bucket.items():
            result.setdefault(code, {})[tp_code] = pop

    log.info("CPV parroquias aggregated: %d", len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: distribute to sectors and upsert
# ─────────────────────────────────────────────────────────────────────────────

def _distribute_and_upsert(
    conn,
    parroquia_pops: dict[str, dict[str, float]],
) -> None:
    """
    For each parroquia in parroquia_pops, find the matching DB census areas
    (where area_code[:6] == parroquia_code), weight by existing all_ages, and
    insert/update census_areas_population.
    """
    with conn.cursor() as cur:
        # Load target_population id map.
        cur.execute("SELECT code, id FROM target_population")
        tp_ids: dict[str, int] = {row[0]: row[1] for row in cur.fetchall()}
        log.info("Target population groups in DB: %s", list(tp_ids.keys()))

        # Load every census area with its id, area_code, and existing all_ages.
        cur.execute("""
            SELECT ca.id,
                   ca.area_code,
                   COALESCE(cap.population, 0.0) AS all_ages_pop
            FROM   census_areas ca
            LEFT JOIN census_areas_population cap
                   ON cap.census_area_id = ca.id
                  AND cap.target_population_id = %(all_ages_id)s
        """, {"all_ages_id": tp_ids["all_ages"]})
        rows = cur.fetchall()

    log.info("Census areas loaded from DB: %d", len(rows))

    # Group census areas by their 6-digit parroquia prefix.
    # sectors_by_parroquia: {parroquia_code: [(area_id, all_ages_weight), ...]}
    sectors_by_parroquia: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for area_id, area_code, all_ages_pop in rows:
        prefix = area_code[:6]
        sectors_by_parroquia[prefix].append((area_id, float(all_ages_pop)))

    log.info("DB parroquias (unique 6-digit prefixes): %d", len(sectors_by_parroquia))

    # Build rows to insert.
    insert_rows: list[tuple[int, int, float]] = []
    unmatched_parroquias = 0

    for parroquia_code, group_pops in parroquia_pops.items():
        sectors = sectors_by_parroquia.get(parroquia_code)
        if not sectors:
            unmatched_parroquias += 1
            continue

        # Compute weight for each sector within this parroquia.
        total_weight = sum(w for _, w in sectors)
        if total_weight <= 0.0:
            # No existing all_ages data — distribute equally.
            n = len(sectors)
            weights = [(area_id, 1.0 / n) for area_id, _ in sectors]
        else:
            weights = [(area_id, w / total_weight) for area_id, w in sectors]

        for tp_code, parroquia_total in group_pops.items():
            tp_id = tp_ids.get(tp_code)
            if tp_id is None:
                continue
            for area_id, fraction in weights:
                insert_rows.append((area_id, tp_id, parroquia_total * fraction))

    if unmatched_parroquias:
        log.warning(
            "%d CPV parroquia codes had no matching census areas "
            "(area_code[:6] not found in DB).",
            unmatched_parroquias,
        )

    if not insert_rows:
        log.error("No rows to insert — verify area_code format and DB contents.")
        return

    log.info("Rows to upsert: %d", len(insert_rows))

    BATCH = 5_000
    total = 0
    with conn.cursor() as cur:
        for i in range(0, len(insert_rows), BATCH):
            execute_values(
                cur,
                """
                INSERT INTO census_areas_population
                    (census_area_id, target_population_id, population)
                VALUES %s
                ON CONFLICT (census_area_id, target_population_id)
                DO UPDATE SET population = EXCLUDED.population
                """,
                insert_rows[i : i + BATCH],
            )
            total += len(insert_rows[i : i + BATCH])
    conn.commit()
    log.info("Upserted %d census_areas_population rows.", total)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--csv", default=DEFAULT_CSV,
        help="Path to CPV2010S_CSV_Poblacion_1.csv (or .zip)",
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help="PostgreSQL connection string",
    )
    args = parser.parse_args()

    parroquia_pops = _aggregate_by_parroquia(args.csv)

    sample = list(parroquia_pops.items())[:3]
    for code, groups in sample:
        log.info("Sample parroquia %s: %s", code, groups)

    db_url = args.db.replace("postgresql+asyncpg://", "postgresql://")
    log.info("Connecting to DB: %s", db_url)
    conn = psycopg2.connect(db_url)
    try:
        _distribute_and_upsert(conn, parroquia_pops)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    log.info("Done.")


if __name__ == "__main__":
    main()
