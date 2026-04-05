"""
Compute and store the average neighbourhood travel speed (vpd) for every
census area.

Algorithm
---------
For each area *a* and every stored neighbour *b* in distance_matrix:

    geo_dist_km  = flat-earth distance between a and b (km)
    speed_km_h   = geo_dist_km / (travel_time_minutes(a, b) / 60)

Step 1 – neighbours within 2 km radius:
    Collect all neighbours whose geo_dist_km <= RADIUS_KM.
    vpd = harmonic mean of their speeds
        = sum(geo_dist_km_i) / sum(travel_time_h_i)
    (Equivalent to total distance / total time — the physically correct
    way to average speeds.  Arithmetic mean over-weights short, fast trips
    and can inflate vpd by 20-40 % in heterogeneous road networks.)

Step 2 – fallback when no neighbours are within the radius:
    Take the FALLBACK_K nearest neighbours by geo distance and apply the
    same harmonic-mean formula.

Step 3 – fallback when the area has no distance_matrix entries at all:
    Assign FALLBACK_SPEED_KMH (default 30 km/h).

The result is written to census_areas.avg_speed_kmh.  The script is
idempotent: it skips areas that already have a value unless --force is used.

Usage
-----
    python compute_avg_speed.py \\
        [--db  postgresql://lip2:lip2@localhost:5432/lip2_ecuador] \\
        [--radius  2.0]   \\
        [--fallback-k 10] \\
        [--fallback 30.0] \\
        [--force]
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from collections import defaultdict

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DB       = os.getenv("DATABASE_URL", "postgresql://lip2:lip2@localhost:5432/lip2_ecuador")
RADIUS_KM        = 2.0    # km — neighbour radius for vpd computation
FALLBACK_K       = 10     # number of nearest neighbours when radius yields none
FALLBACK_SPEED   = 30.0   # km/h — used when an area has no distance_matrix entries
BATCH_SIZE       = 5_000  # rows per UPDATE batch


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def _geo_dist_km(x_a: float, y_a: float, x_b: float, y_b: float) -> float:
    """Flat-earth Euclidean distance in km (valid for local scale)."""
    lat_mid_rad = math.radians((y_a + y_b) / 2.0)
    dx_km = (x_b - x_a) * 111.0 * math.cos(lat_mid_rad)
    dy_km = (y_b - y_a) * 111.0
    return math.sqrt(dx_km * dx_km + dy_km * dy_km)


def _harmonic_mean_speed(pairs: list[tuple[float, float]]) -> float:
    """
    Compute vpd as harmonic mean of speeds from (geo_dist_km, travel_time_h) pairs.

    harmonic_mean = total_distance / total_time
                  = sum(d_i) / sum(d_i / speed_i)
                  = sum(d_i) / sum(t_i)
    """
    total_dist = 0.0
    total_time = 0.0
    for d_km, t_h in pairs:
        total_dist += d_km
        total_time += t_h
    if total_time <= 0.0:
        return 0.0
    return total_dist / total_time


def _compute_speeds(
    conn,
    radius_km: float,
    fallback_k: int,
    fallback_speed: float,
    force: bool,
) -> dict[int, float]:
    """
    Return {area_id: vpd_km_h} for all areas.

    vpd is computed as the harmonic mean of travel speeds to neighbours
    within `radius_km`.  If no neighbours fall within that radius, the
    `fallback_k` nearest neighbours (by geographic distance) are used.
    Areas with no distance_matrix entries at all receive `fallback_speed`.
    """
    log.info("Loading census area coordinates …")
    with conn.cursor() as cur:
        if force:
            cur.execute("SELECT id, x, y FROM census_areas WHERE x IS NOT NULL AND y IS NOT NULL")
        else:
            cur.execute(
                "SELECT id, x, y FROM census_areas "
                "WHERE x IS NOT NULL AND y IS NOT NULL AND avg_speed_kmh IS NULL"
            )
        area_rows = cur.fetchall()

    if not area_rows:
        log.info("No areas to process (all already populated; use --force to recompute).")
        return {}

    area_ids = [r[0] for r in area_rows]
    area_xy  = {r[0]: (float(r[1]), float(r[2])) for r in area_rows}
    log.info("Areas to process: %d", len(area_ids))

    # Load distance_matrix entries for target areas.
    log.info("Loading distance_matrix entries …")
    id_list = ",".join(str(i) for i in area_ids)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT dm.from_area_id,
                   dm.travel_time_minutes,
                   ca2.x  AS x_to,
                   ca2.y  AS y_to
            FROM   distance_matrix dm
            JOIN   census_areas ca2 ON ca2.id = dm.to_area_id
            WHERE  dm.from_area_id IN ({id_list})
              AND  dm.travel_time_minutes > 0
              AND  ca2.x IS NOT NULL
              AND  ca2.y IS NOT NULL
        """)
        dm_rows = cur.fetchall()

    log.info("Distance-matrix rows loaded: %d", len(dm_rows))

    # Build per-area list of (geo_dist_km, travel_time_h) pairs.
    # Each tuple carries enough info to compute speed and to sort by distance.
    neighbours: dict[int, list[tuple[float, float]]] = defaultdict(list)

    for from_id, tt_min, x_to, y_to in dm_rows:
        if from_id not in area_xy:
            continue
        x_a, y_a = area_xy[from_id]
        d_km = _geo_dist_km(x_a, y_a, float(x_to), float(y_to))
        if d_km <= 0.0:
            continue
        t_h = float(tt_min) / 60.0
        if t_h <= 0.0:
            continue
        neighbours[from_id].append((d_km, t_h))

    # Compute vpd for each area.
    result: dict[int, float] = {}
    no_radius = 0
    no_data   = 0

    for area_id in area_ids:
        pairs = neighbours.get(area_id)
        if not pairs:
            # No distance_matrix entries at all.
            result[area_id] = fallback_speed
            no_data += 1
            continue

        # Step 1: neighbours within radius_km.
        within = [(d, t) for d, t in pairs if d <= radius_km]

        if within:
            result[area_id] = _harmonic_mean_speed(within)
        else:
            # Step 2: fallback_k nearest neighbours by geo distance.
            no_radius += 1
            pairs_sorted = sorted(pairs, key=lambda p: p[0])
            nearest_k    = pairs_sorted[:fallback_k]
            spd = _harmonic_mean_speed(nearest_k)
            result[area_id] = spd if spd > 0.0 else fallback_speed

    log.info(
        "vpd computed: %d areas  |  %d had no neighbours within %.1f km (used %d-nearest fallback)"
        "  |  %d used default %.1f km/h",
        len(result), no_radius, radius_km, fallback_k, no_data, fallback_speed,
    )
    return result


def _persist(conn, speeds: dict[int, float]) -> None:
    """Bulk-update census_areas.avg_speed_kmh in batches."""
    rows  = list(speeds.items())
    total = 0
    with conn.cursor() as cur:
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]
            execute_values(
                cur,
                """
                UPDATE census_areas AS ca
                SET    avg_speed_kmh = data.speed
                FROM   (VALUES %s) AS data(id, speed)
                WHERE  ca.id = data.id
                """,
                batch,
                template="(%s, %s)",
            )
            total += len(batch)
            log.info("  Updated %d / %d areas …", total, len(rows))
    conn.commit()
    log.info("Done. avg_speed_kmh (vpd) populated for %d areas.", total)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db",         default=DEFAULT_DB,
                        help="PostgreSQL connection string")
    parser.add_argument("--radius",     type=float, default=RADIUS_KM,
                        help=f"Neighbour radius in km (default {RADIUS_KM})")
    parser.add_argument("--fallback-k", type=int,   default=FALLBACK_K,
                        help=f"Nearest neighbours used when radius yields none "
                             f"(default {FALLBACK_K})")
    parser.add_argument("--fallback",   type=float, default=FALLBACK_SPEED,
                        help=f"Speed (km/h) for areas with no distance_matrix entry "
                             f"(default {FALLBACK_SPEED})")
    parser.add_argument("--force",      action="store_true",
                        help="Recompute even for areas that already have avg_speed_kmh set")
    args = parser.parse_args()

    db_url = args.db.replace("postgresql+asyncpg://", "postgresql://")
    log.info("Connecting to %s …", db_url)
    conn = psycopg2.connect(db_url)

    try:
        speeds = _compute_speeds(conn, args.radius, args.fallback_k, args.fallback, args.force)
        if speeds:
            _persist(conn, speeds)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
