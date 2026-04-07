"""
Update facility capacity in lip2_belgium database.

Data sources:
  - Hospitals:      FPS Public Health Belgium - "Erkende en verantwoorde bedden 2-2023"
                    https://www.health.belgium.be (XLSX downloaded and parsed)
  - Schools:        Statbel / Onderwijs Vlaanderen national averages 2022-2023
                    Primary+secondary avg: ~350 students per school
  - High schools:   Belgian hogeschool/university campus avg: ~2000 students
  - Health centers: RIZIV standard - avg GP practice patient list: ~1800 patients

Usage:
  python scripts/update_belgium_capacity.py [--dry-run]

  --dry-run   Print SQL without executing it.
"""

import argparse
import sys
import urllib.request
import tempfile
import os

# ── Hospital capacity: manually verified matches from FPS Health XLSX ────────
#
# Source: FPS Public Health Belgium
#   "Erkende en verantwoorde bedden 2-2023" (recognized beds, 2nd semester 2023)
#   URL: https://www.health.belgium.be/.../verantwoorde_erkend_bedden_2-2023.xlsx
#
# Match method: fuzzy name matching + manual verification.
# Only high-confidence matches are included. Others use conservative estimates.
#
# Format: facility_id → (beds, source, confidence)
HOSPITAL_CAPACITY = {
    # ── FPS XLSX direct matches ─────────────────────────────────────────────
    2398: (438,  "FPS XLSX - A.Z. ST. BLASIUS",                             "high"),
    654:  (159,  "FPS XLSX - CENTRE NEUROLOGIQUE WILLIAM LENNOX",           "high"),
    3110: (313,  "FPS XLSX - PSYCHIATRISCH ZIEKENHUIS STUIVENBERG",         "high"),

    # ── FPS XLSX matches with network-level aggregation ─────────────────────
    6731: (440,  "FPS XLSX - ISoSL Petit Bourgogne - Agora (one campus)",   "medium"),
    5282: (250,  "FPS XLSX - C.H. EPICURA Beloeil area estimate",           "medium"),

    # ── Estimates based on facility type and Belgian national benchmarks ─────
    # Source: SPF Sante annual reports, healthybelgium.be
    258:  (240,  "Estimate - major pediatric university hospital Brussels",  "estimate"),
    3113: (120,  "Estimate - ZAS children's hospital Antwerp",              "estimate"),
    3116: (65,   "Estimate - child & youth psychiatry unit Antwerp",        "estimate"),
    1906: (44,   "Estimate - psychosocial center (small psychiatric unit)",  "estimate"),
    5277: (36,   "Estimate - day psychotherapy center",                     "estimate"),
    3359: (30,   "Estimate - sports medicine clinic",                       "estimate"),
    1486: (50,   "Estimate - small private clinic",                         "estimate"),
    742:  (80,   "Estimate - palliative/religious care facility",            "estimate"),
    71:   (50,   "Estimate - medical center (outpatient focus)",             "estimate"),
    2913: (60,   "Estimate - MS rehabilitation center",                     "estimate"),
    2498: (30,   "Estimate - university student health center",              "estimate"),
    3361: (50,   "Estimate - Red Cross care facility",                      "estimate"),
    1547: (20,   "Estimate - small specialist clinic",                      "estimate"),
}

# ── National average capacities ──────────────────────────────────────────────
#
# school (1080 facilities):
#   Source: Statbel / Onderwijs Vlaanderen 2022-2023
#   Belgian primary school avg: ~260 students; secondary: ~540 students.
#   Combined avg across both levels: ~350 students.
#
# high_school (142 facilities):
#   These are Belgian hogescholen and university campuses, not secondary schools.
#   Source: VLIR / CRef Belgian higher education statistics 2022-2023.
#   Average enrollment per campus: ~2000 students.
#
# health_center (5954 facilities):
#   Individual GP practices (huisartspraktijk / cabinet de medecin generaliste).
#   Source: RIZIV/INAMI - average Belgian GP patient list: ~1800 patients.
#
AVG_CAPACITY = {
    "school":        350,
    "high_school":  2000,
    "health_center": 1800,
}


def build_sql(dry_run: bool) -> list[str]:
    statements = []

    # Hospitals - individual capacity per facility
    statements.append("-- === HOSPITALS (FPS Health XLSX + estimates) ===")
    for fid, (beds, source, confidence) in HOSPITAL_CAPACITY.items():
        statements.append(
            f"-- [{confidence}] {source}\n"
            f"UPDATE facilities SET capacity = {beds} WHERE id = {fid};"
        )

    # Other types - uniform national average
    statements.append("\n-- === SCHOOLS, HIGH SCHOOLS, HEALTH CENTERS (national averages) ===")
    for ftype, avg in AVG_CAPACITY.items():
        statements.append(
            f"-- {ftype}: Belgian national average = {avg}\n"
            f"UPDATE facilities SET capacity = {avg}\n"
            f"  WHERE facility_type = '{ftype}' AND status = 'existing' AND capacity = 0;"
        )

    return statements


def run(dry_run: bool) -> None:
    statements = build_sql(dry_run)
    sql_block = "\n".join(statements)

    if dry_run:
        print("=== DRY RUN — SQL to be executed ===\n")
        print(sql_block)
        return

    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(
        host="localhost", port=5432,
        dbname="lip2_belgium", user="lip2", password="lip2",
    )
    cur = conn.cursor()

    total_updated = 0
    for stmt in statements:
        if not stmt.strip().startswith("UPDATE"):
            continue
        cur.execute(stmt)
        total_updated += cur.rowcount

    conn.commit()
    cur.close()
    conn.close()
    print(f"Done. {total_updated} rows updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update facility capacity in lip2_belgium.")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing.")
    args = parser.parse_args()
    run(args.dry_run)
