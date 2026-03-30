"""
Speed histogram comparison between Ecuador and Belgium.

For every pair (from_area, to_area) in distance_matrix the implicit speed is:

    speed_km_h = euclidean_km(from, to) / (travel_time_minutes / 60)

where euclidean_km uses a flat-earth approximation from WGS-84 lon/lat:

    D_km = sqrt( ((x_b - x_a) * 111 * cos(lat_mid_rad))^2
               + ((y_b - y_a) * 111)^2 )

Pairs with travel_time = 0 or identical coordinates are skipped.
The top 0.1% of speeds (outliers) are excluded from the plot but counted.

Usage
-----
    python speed_histogram_comparison.py \\
        [--db-ecuador postgresql://lip2:lip2@localhost:5432/lip2_ecuador] \\
        [--db-belgium postgresql://lip2:lip2@localhost:5432/lip2_belgium] \\
        [--sample 500000] \\
        [--out speed_histogram_comparison.png]
"""

from __future__ import annotations

import argparse
import logging
import math
import os

import numpy as np
import psycopg2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Speed colour bands (same as the reference histogram).
BANDS = [
    (0,   20,  "#e74c3c", "< 20 km/h  (peatonal / trocha)"),
    (20,  50,  "#e67e22", "20–50 km/h  (vía secundaria)"),
    (50,  90,  "#27ae60", "50–90 km/h  (carretera normal)"),
    (90,  150, "#2980b9", "90–150 km/h  (vía rápida)"),
]
CLIP_MAX = 150   # km/h — speeds above this are treated as outliers for display


def _fetch_speeds(conn_str: str, sample: int | None) -> np.ndarray:
    """
    Query distance_matrix joined with census_areas coordinates and return
    an array of implicit speeds (km/h).  Pairs with zero travel time or
    zero distance are silently dropped.
    """
    limit_clause = f"LIMIT {sample}" if sample else ""
    sql = f"""
        SELECT
            fa.x  AS x_a,
            fa.y  AS y_a,
            fb.x  AS x_b,
            fb.y  AS y_b,
            dm.travel_time_minutes
        FROM distance_matrix dm
        JOIN census_areas fa ON fa.id = dm.from_area_id
        JOIN census_areas fb ON fb.id = dm.to_area_id
        WHERE dm.travel_time_minutes > 0
          AND fa.x IS NOT NULL AND fa.y IS NOT NULL
          AND fb.x IS NOT NULL AND fb.y IS NOT NULL
          AND (fa.x <> fb.x OR fa.y <> fb.y)
        {limit_clause}
    """
    log.info("Connecting to %s", conn_str.split("@")[-1])
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            log.info("Executing query (may take a moment)…")
            cur.execute(sql)
            rows = cur.fetchall()

    log.info("  %d pairs fetched", len(rows))

    x_a = np.array([r[0] for r in rows], dtype=np.float64)
    y_a = np.array([r[1] for r in rows], dtype=np.float64)
    x_b = np.array([r[2] for r in rows], dtype=np.float64)
    y_b = np.array([r[3] for r in rows], dtype=np.float64)
    tt  = np.array([r[4] for r in rows], dtype=np.float64)

    lat_mid = np.radians((y_a + y_b) / 2.0)
    dx_km   = (x_b - x_a) * 111.0 * np.cos(lat_mid)
    dy_km   = (y_b - y_a) * 111.0
    d_km    = np.sqrt(dx_km ** 2 + dy_km ** 2)

    # Drop zero-distance pairs.
    valid  = (d_km > 1e-6) & (tt > 0)
    d_km   = d_km[valid]
    tt     = tt[valid]

    speeds = d_km / (tt / 60.0)
    log.info("  %d valid speed values (min=%.1f  max=%.1f  median=%.1f km/h)",
             len(speeds), speeds.min(), speeds.max(), np.median(speeds))
    return speeds


def _band_color(speed_val: float) -> str:
    for lo, hi, color, _ in BANDS:
        if lo <= speed_val < hi:
            return color
    return "#95a5a6"  # grey for anything outside bands


def _plot_histogram(
    ax: plt.Axes,
    speeds: np.ndarray,
    title: str,
    bin_width: float = 2.0,
) -> None:
    """Draw a colour-banded histogram on `ax`."""
    outliers = int(np.sum(speeds > CLIP_MAX))
    speeds_clipped = speeds[speeds <= CLIP_MAX]

    bins = np.arange(0, CLIP_MAX + bin_width, bin_width)
    counts, edges = np.histogram(speeds_clipped, bins=bins)

    n_total  = len(speeds)
    p5, p95  = np.percentile(speeds_clipped, [5, 95])
    median   = float(np.median(speeds_clipped))

    for i, (count, left) in enumerate(zip(counts, edges[:-1])):
        color = _band_color(left + bin_width / 2)
        ax.bar(left, count, width=bin_width, color=color, align="edge",
               edgecolor="white", linewidth=0.3)

    ax.set_xlim(0, CLIP_MAX)
    ax.set_xlabel("Velocidad de transporte (km/h)", fontsize=11)
    ax.set_ylabel("Número de pares", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    subtitle = (
        f"n = {n_total:,} pares  │  mediana = {median:.1f} km/h  │  "
        f"p5–p95 = {p5:.1f}–{p95:.1f} km/h"
    )
    if outliers:
        subtitle += f"  │  {outliers:,} pares > {CLIP_MAX} km/h (no mostrados)"

    ax.set_title(f"{title}\n{subtitle}", fontsize=12, pad=8)

    # Legend patches.
    patches = [
        mpatches.Patch(color=color, label=label)
        for _, _, color, label in BANDS
    ]
    ax.legend(handles=patches, fontsize=9, loc="upper right")


def main() -> None:
    parser = argparse.ArgumentParser(description="Speed histogram comparison Ecuador vs Belgium")
    parser.add_argument(
        "--db-ecuador",
        default=os.getenv("DB_ECUADOR", "postgresql://lip2:lip2@localhost:5432/lip2_ecuador"),
    )
    parser.add_argument(
        "--db-belgium",
        default=os.getenv("DB_BELGIUM", "postgresql://lip2:lip2@localhost:5432/lip2_belgium"),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit the number of rows fetched per database (None = all pairs).",
    )
    parser.add_argument(
        "--out",
        default="speed_histogram_comparison.png",
        help="Output file path.",
    )
    args = parser.parse_args()

    log.info("=== Ecuador ===")
    speeds_ec = _fetch_speeds(args.db_ecuador, args.sample)

    log.info("=== Belgium ===")
    speeds_be = _fetch_speeds(args.db_belgium, args.sample)

    fig, (ax_ec, ax_be) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        "Velocidades implícitas en distance_matrix — Ecuador vs Bélgica",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    _plot_histogram(ax_ec, speeds_ec, "Ecuador")
    _plot_histogram(ax_be, speeds_be, "Bélgica")

    plt.tight_layout()
    out_path = args.out
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
