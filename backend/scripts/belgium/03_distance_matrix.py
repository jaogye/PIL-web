"""
Belgium ETL – Step 3: Compute sparse distance matrix via OSRM.

Strategy:
  1. Load all census area centroids from the DB.
  2. Build a KD-tree; for each centroid find its 600 geographic nearest
     neighbours (Euclidean, lat/lon units – good enough for neighbour search).
  3. Process centroids in batches of BATCH_SOURCES.
     For each batch:
       a. Collect unique coordinates (sources + all their candidates).
       b. POST to OSRM /table → get travel times for sources × all candidates.
       c. For each source keep the ≤ MAX_NEIGHBORS pairs with the smallest time.
  4. Bulk-insert results into distance_matrix.

Prerequisites:
  OSRM must be running on OSRM_URL (default http://localhost:5000).
  See README.md for Docker setup instructions.

Usage:
    python 03_distance_matrix.py [options]

    --db         PostgreSQL connection URL (lip2_belgium)
    --osrm       OSRM base URL (default http://localhost:5000)
    --neighbors  Max neighbors per area to store (default 500)
    --candidates Geo-nearest candidates to evaluate per area (default 600)
    --batch      OSRM sources per request (default 50)
    --max-time   Maximum travel time in minutes to store (default 180)
    --resume     Skip already-computed from_area_ids
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from typing import Any

import numpy as np
import psycopg2
import requests
from psycopg2.extras import execute_values
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DB   = os.getenv("DATABASE_URL_BELGIUM",
                         "postgresql://lip2:lip2@localhost:5432/lip2_belgium")
DEFAULT_OSRM = os.getenv("OSRM_URL", "http://localhost:5000")

BATCH_SOURCES  = 1     # sources per OSRM /table call (keep URLs short)
MAX_NEIGHBORS  = 500   # pairs to store per source area
GEO_CANDIDATES = 550   # geographic nearest neighbours to evaluate
MAX_TIME_MIN   = 180.0 # discard pairs with travel time > this


# ─────────────────────────────────────────────────────────────────────────────
# OSRM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _osrm_table(coords: list[tuple[float, float]],
                source_indices: list[int],
                dest_indices: list[int],
                osrm_base: str,
                retries: int = 3) -> list[list[float | None]] | None:
    """
    Call OSRM /table/v1/driving endpoint.
    coords: list of (lon, lat)
    Returns matrix[src][dst] = travel_time_seconds or None on failure.
    """
    coord_str = ";".join(f"{lon},{lat}" for lon, lat in coords)
    src_str   = ";".join(str(i) for i in source_indices)
    dst_str   = ";".join(str(i) for i in dest_indices)
    url = (f"{osrm_base}/table/v1/driving/{coord_str}"
           f"?sources={src_str}&destinations={dst_str}&annotations=duration")

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
            if data.get("code") != "Ok":
                log.warning("OSRM error: %s", data.get("message", data.get("code")))
                return None
            return data["durations"]
        except requests.exceptions.RequestException as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt
                log.warning("  OSRM request failed (%s); retrying in %ds …", exc, wait)
                time.sleep(wait)
            else:
                log.error("  OSRM request failed after %d attempts: %s", retries, exc)
                return None
    return None


def check_osrm(osrm_base: str) -> bool:
    """Verify OSRM is reachable with a simple health request."""
    try:
        # Tiny table query: Brussels centre
        r = requests.get(
            f"{osrm_base}/table/v1/driving/4.35,50.85;4.36,50.86?sources=0&destinations=1",
            timeout=10)
        return r.status_code == 200 and r.json().get("code") == "Ok"
    except Exception as exc:
        log.error("OSRM not reachable at %s: %s", osrm_base, exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",         default=DEFAULT_DB)
    parser.add_argument("--osrm",       default=DEFAULT_OSRM)
    parser.add_argument("--neighbors",  type=int,   default=MAX_NEIGHBORS)
    parser.add_argument("--candidates", type=int,   default=GEO_CANDIDATES)
    parser.add_argument("--batch",      type=int,   default=BATCH_SOURCES)
    parser.add_argument("--max-time",   type=float, default=MAX_TIME_MIN)
    parser.add_argument("--resume",     action="store_true",
                        help="Skip from_area_ids already in distance_matrix")
    args = parser.parse_args()

    # ── Check OSRM ──
    log.info("Checking OSRM at %s …", args.osrm)
    if not check_osrm(args.osrm):
        log.error(
            "OSRM is not running.\n"
            "Start it with Docker (see README.md):\n"
            "  docker run -t -p 5000:5000 -v /tmp/osrm:/data "
            "osrm/osrm-backend osrm-routed --algorithm mld /data/belgium.osrm"
        )
        sys.exit(1)
    log.info("OSRM OK.")

    # ── Load centroids ──
    db_url = args.db.replace("postgresql+asyncpg://", "postgresql://")

    def make_conn():
        c = psycopg2.connect(
            db_url,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5,
        )
        c.autocommit = False
        return c

    conn = make_conn()

    with conn.cursor() as cur:
        cur.execute("SELECT id, x, y FROM census_areas WHERE x IS NOT NULL AND y IS NOT NULL ORDER BY id")
        rows = cur.fetchall()

    area_ids = [r[0] for r in rows]
    coords   = np.array([[r[1], r[2]] for r in rows], dtype=np.float64)  # (N, 2) lon/lat
    n = len(area_ids)
    log.info("Loaded %d census area centroids.", n)

    # ── Resume: find already-computed sources ──
    done_set: set[int] = set()
    if args.resume:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT from_area_id FROM distance_matrix")
            done_set = {r[0] for r in cur.fetchall()}
        log.info("Resume: %d sources already done.", len(done_set))

    # ── Build KD-tree ──
    log.info("Building KD-tree …")
    tree = cKDTree(coords)
    k = min(args.candidates + 1, n)  # +1 because the point itself is returned at dist=0
    log.info("Querying %d nearest geographic neighbours for each centroid …", k - 1)
    _, neighbour_idx = tree.query(coords, k=k)
    # neighbour_idx[i] = sorted indices of k nearest (first is i itself)

    # ── Process in batches ──
    idx_to_pos = {aid: i for i, aid in enumerate(area_ids)}
    total_pairs = 0
    t_start = time.time()
    n_areas = len(area_ids)

    for batch_start in range(0, n_areas, args.batch):
        batch_end = min(batch_start + args.batch, n_areas)
        source_positions = list(range(batch_start, batch_end))

        # Skip already-done sources
        if args.resume:
            source_positions = [p for p in source_positions if area_ids[p] not in done_set]
        if not source_positions:
            continue

        # Collect all unique candidate positions for this batch
        cand_set: set[int] = set(source_positions)
        for sp in source_positions:
            for ni in neighbour_idx[sp]:
                if ni != sp:
                    cand_set.add(int(ni))

        cand_list = sorted(cand_set)
        pos_to_local = {pos: i for i, pos in enumerate(cand_list)}
        local_coords  = [tuple(coords[pos]) for pos in cand_list]
        local_sources = [pos_to_local[sp] for sp in source_positions]
        local_dests   = list(range(len(cand_list)))

        # OSRM table call
        durations = _osrm_table(local_coords, local_sources, local_dests,
                                args.osrm)
        if durations is None:
            log.warning("Skipping batch %d–%d (OSRM error)", batch_start, batch_end)
            continue

        # Build insert rows
        insert_rows = []
        for li, sp in enumerate(source_positions):
            from_id = area_ids[sp]
            row_times = durations[li]  # list of seconds or None for each dest

            pairs: list[tuple[float, int]] = []
            for lj, dest_pos in enumerate(cand_list):
                if dest_pos == sp:
                    continue
                t = row_times[lj]
                if t is None or t <= 0:
                    continue
                t_min = t / 60.0
                if t_min > args.max_time:
                    continue
                pairs.append((t_min, area_ids[dest_pos]))

            # Sort by travel time, keep top N
            pairs.sort()
            for t_min, to_id in pairs[:args.neighbors]:
                insert_rows.append((from_id, to_id, round(t_min, 4)))

        if insert_rows:
            for attempt in range(3):
                try:
                    with conn.cursor() as cur:
                        execute_values(cur, """
                            INSERT INTO distance_matrix (from_area_id, to_area_id, travel_time_minutes)
                            VALUES %s
                            ON CONFLICT (from_area_id, to_area_id) DO UPDATE
                                SET travel_time_minutes = EXCLUDED.travel_time_minutes
                        """, insert_rows)
                    conn.commit()
                    total_pairs += len(insert_rows)
                    break
                except psycopg2.OperationalError as exc:
                    log.warning("DB connection lost (%s); reconnecting …", exc)
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = make_conn()

        elapsed = time.time() - t_start
        done    = batch_end
        pct     = done * 100 / n_areas
        eta_s   = (elapsed / done) * (n_areas - done) if done > 0 else 0
        log.info("  %d/%d (%.1f%%)  pairs_inserted=%d  elapsed=%.0fs  ETA=%.0fs",
                 done, n_areas, pct, total_pairs, elapsed, eta_s)

    conn.close()
    log.info("Distance matrix complete.  Total pairs: %d", total_pairs)


if __name__ == "__main__":
    main()
