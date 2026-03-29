"""
Compute and store the median neighbourhood speed for every census area.

For each area *a* the script calculates, for every stored neighbour *b*:

    speed_km_h = euclidean_km(a, b) / (travel_time_minutes(a, b) / 60)

where euclidean_km is derived from WGS-84 lon/lat coordinates using a
flat-earth approximation valid at local scale:

    D_km = sqrt( ((x_b - x_a) * 111 * cos(lat_mid_rad))^2
               + ((y_b - y_a) * 111)^2 )

The MEDIAN of all neighbour speeds is then stored in
census_areas.avg_speed_kmh (robust against highway outliers).

Areas with no valid neighbours receive a configurable fallback speed
(default 30 km/h).

The script runs against any lip2_* database and can be resumed safely
(it skips areas already populated unless --force is passed).

Usage
-----
    python compute_avg_speed.py \\
        [--db  postgresql://lip2:lip2@localhost:5432/lip2_ecuador] \\
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
FALLBACK_SPEED   = 30.0   # km/h — used when an area has no valid neighbours
BATCH_SIZE       = 5_000  # rows per UPDATE batch


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_speeds(
    conn,
    fallback_speed: float,
    force: bool,
) -> dict[int, float]:
    """
    Return {area_id: median_speed_km_h} for all areas that have at least
    one valid distance_matrix entry.  Areas without entries receive
    `fallback_speed`.
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

    area_ids   = [r[0] for r in area_rows]
    area_xy    = {r[0]: (float(r[1]), float(r[2])) for r in area_rows}
    id_set     = set(area_ids)
    log.info("Areas to process: %d", len(area_ids))

    # Load distance_matrix entries where the source area needs recomputation.
    # Include the destination coordinates via a JOIN.
    log.info("Loading distance_matrix entries …")
    id_list = ",".join(str(i) for i in area_ids)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT dm.from_area_id,
                   dm.travel_time_minutes,
                   ca2.x AS x_to,
                   ca2.y AS y_to
            FROM   distance_matrix dm
            JOIN   census_areas ca2 ON ca2.id = dm.to_area_id
            WHERE  dm.from_area_id IN ({id_list})
              AND  dm.travel_time_minutes > 0
              AND  ca2.x IS NOT NULL
              AND  ca2.y IS NOT NULL
        """)
        dm_rows = cur.fetchall()

    log.info("Distance-matrix rows loaded: %d", len(dm_rows))

    # Group speeds per source area.
    speeds_per_area: dict[int, list[float]] = defaultdict(list)

    for from_id, tt_min, x_to, y_to in dm_rows:
        if from_id not in area_xy:
            continue
        x_from, y_from = area_xy[from_id]
        lat_mid_rad = math.radians((y_from + float(y_to)) / 2.0)
        dx_km = (float(x_to) - x_from) * 111.0 * math.cos(lat_mid_rad)
        dy_km = (float(y_to) - y_from) * 111.0
        d_km  = math.sqrt(dx_km * dx_km + dy_km * dy_km)
        if d_km <= 0.0:
            continue
        speed = d_km / (float(tt_min) / 60.0)
        if speed > 0.0:
            speeds_per_area[from_id].append(speed)

    # Compute median per area; assign fallback when no data.
    result: dict[int, float] = {}
    for area_id in area_ids:
        vals = speeds_per_area.get(area_id, [])
        if vals:
            result[area_id] = float(np.median(vals))
        else:
            result[area_id] = fallback_speed

    no_data = sum(1 for aid in area_ids if not speeds_per_area.get(aid))
    log.info(
        "Speeds computed: %d areas  |  %d used fallback %.1f km/h",
        len(result), no_data, fallback_speed,
    )
    return result


def _persist(conn, speeds: dict[int, float]) -> None:
    """Bulk-update census_areas.avg_speed_kmh in batches."""
    rows = list(speeds.items())   # [(area_id, speed), …]
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
    log.info("Done. avg_speed_kmh populated for %d areas.", total)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db",       default=DEFAULT_DB,
                        help="PostgreSQL connection string")
    parser.add_argument("--fallback", type=float, default=FALLBACK_SPEED,
                        help="Speed (km/h) for areas with no distance_matrix entry "
                             f"(default {FALLBACK_SPEED})")
    parser.add_argument("--force",    action="store_true",
                        help="Recompute even for areas that already have avg_speed_kmh set")
    args = parser.parse_args()

    db_url = args.db.replace("postgresql+asyncpg://", "postgresql://")
    log.info("Connecting to %s …", db_url)
    conn = psycopg2.connect(db_url)

    try:
        speeds = _compute_speeds(conn, args.fallback, args.force)
        if speeds:
            _persist(conn, speeds)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
