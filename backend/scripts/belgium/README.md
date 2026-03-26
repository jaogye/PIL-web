# Belgium ETL Setup Guide

This guide populates the `lip2_belgium` database with Belgian census data, facilities, and a travel-time distance matrix.

## Prerequisites

- Docker Desktop (with the `lip2-web-db-1` container running)
- Python 3.10+ with packages: `geopandas`, `psycopg2-binary`, `requests`, `scipy`, `pyogrio`
- ~5 GB free disk space (OSM data + OSRM processed files)
- ~2 GB RAM for OSRM routing

---

## Step 0 – Create the database and schema

```bash
# From the project root
docker exec -i lip2-web-db-1 psql -U lip2 -c "CREATE DATABASE lip2_belgium OWNER lip2;"
docker exec -i lip2-web-db-1 psql -U lip2 -d lip2_belgium \
  -f /dev/stdin < backend/scripts/belgium/00_schema.sql
```

Verify:
```bash
docker exec lip2-web-db-1 psql -U lip2 -d lip2_belgium -c "\dt"
```

---

## Step 1 – Load census areas and political divisions

Downloads automatically from Statbel (open data, CC BY 4.0):
- Statistical sectors 2024 (~60 MB ZIP with GeoJSON, EPSG:3812 → reprojected to WGS84)
- Population by sector 2024 (~5 MB ZIP with pipe-delimited TXT)

```bash
cd backend/scripts/belgium
python 01_load_census_areas.py
```

Expected output:
```
political_division = 595   (3 regions + 11 provinces + 581 municipalities)
census_areas       = 19,795 (statistical sectors)
total_demand       = ~11,500,000 (population of Belgium)
```

### Manual download fallback

If the Statbel URLs have changed:
1. Visit https://statbel.fgov.be/en/open-data/statistical-sectors-2024
   Download the GeoJSON ZIP and pass:
   ```bash
   python 01_load_census_areas.py --sector-file /path/to/sectors.zip
   ```
2. Visit https://statbel.fgov.be/en/open-data/population-statistical-sector-2024
   Download the TXT ZIP and pass:
   ```bash
   python 01_load_census_areas.py --sector-file ... --pop-file /path/to/population.zip
   ```

---

## Step 2 – Load facilities (schools and hospitals)

Downloads from Geofabrik (OpenStreetMap Belgium free shapefile, ~230 MB ZIP):

```bash
python 02_load_facilities.py
```

Expected output:
```
school         ~3,500
high_school    ~1,200
hospital         ~180
health_center  ~4,000
total          ~7,200
```

### Manual download fallback

```bash
curl -o /tmp/belgium-latest-free.shp.zip \
  https://download.geofabrik.de/europe/belgium-latest-free.shp.zip

python 02_load_facilities.py --shp-zip /tmp/belgium-latest-free.shp.zip
```

---

## Step 3 – Compute distance matrix (OSRM)

### 3a – Download the OSM file

Download the Belgium OSM extract (~650 MB) from Geofabrik and place it in this directory:

```
backend/scripts/belgium/belgium-latest.osm.pbf
```

### 3b – Pre-process with OSRM

Run the three OSRM pre-processing steps (takes ~10–15 min, ~2 GB RAM).

**Important for Windows:** Use `MSYS_NO_PATHCONV=1` to prevent Git Bash from mangling the `/opt/car.lua` path.

```bash
# Extract
MSYS_NO_PATHCONV=1 docker run -t \
  -v "C:/path/to/PIL-web/backend/scripts/belgium:/data" \
  osrm/osrm-backend osrm-extract -p /opt/car.lua /data/belgium-latest.osm.pbf

# Partition
MSYS_NO_PATHCONV=1 docker run -t \
  -v "C:/path/to/PIL-web/backend/scripts/belgium:/data" \
  osrm/osrm-backend osrm-partition /data/belgium-latest.osrm

# Customize
MSYS_NO_PATHCONV=1 docker run -t \
  -v "C:/path/to/PIL-web/backend/scripts/belgium:/data" \
  osrm/osrm-backend osrm-customize /data/belgium-latest.osrm
```

### 3c – Start OSRM server

```bash
MSYS_NO_PATHCONV=1 docker run -d --name osrm-belgium -p 5000:5000 \
  -v "C:/path/to/PIL-web/backend/scripts/belgium:/data" \
  osrm/osrm-backend osrm-routed --algorithm mld /data/belgium-latest.osrm
```

Verify (needs at least two coordinates):
```bash
curl "http://localhost:5000/table/v1/driving/4.35,50.85;4.36,50.86?sources=0&destinations=1"
# Should return {"code":"Ok", ...}
```

### 3d – Run the distance matrix script

```bash
python 03_distance_matrix.py
```

This will take **~2 hours** for 19,795 sectors × 550 nearest neighbours.

Options:
```
--neighbors  500     # pairs stored per area (default)
--candidates 550     # geographic candidates evaluated per area (default)
--batch      1       # OSRM sources per request (keep URLs short; default)
--max-time   180     # max travel time in minutes to store
--resume             # resume safely after interruption
```

Resume after interruption (safe to run multiple times):
```bash
python 03_distance_matrix.py --resume
```

Expected final result:
```
Total pairs: ~9,000,000 – 10,000,000
```

### Stop OSRM when done

```bash
docker stop osrm-belgium && docker rm osrm-belgium
```

---

## Step 4 – Verify the data

```bash
docker exec lip2-web-db-1 psql -U lip2 -d lip2_belgium -c "
  SELECT 'political_division' AS table_name, COUNT(*) FROM political_division
  UNION ALL
  SELECT 'census_areas',    COUNT(*) FROM census_areas
  UNION ALL
  SELECT 'facilities',      COUNT(*) FROM facilities
  UNION ALL
  SELECT 'distance_matrix', COUNT(*) FROM distance_matrix;
"
```

Expected:
```
 table_name        | count
-------------------+---------
 political_division|     595
 census_areas      |  19795
 facilities        |   7194
 distance_matrix   | 9897500  (approximately)
```

Check population:
```bash
docker exec lip2-web-db-1 psql -U lip2 -d lip2_belgium -c "
  SELECT SUM(demand)::bigint AS total_population FROM census_areas;
"
# Expected: ~11,500,000
```

---

## Administrative hierarchy

| Belgium level | DB field | `level` value | Example |
|---|---|---|---|
| Region (3) | `political_division` | `region` | Flemish Region |
| Province (11) | `political_division` | `province` | Antwerpen |
| Municipality (581) | `political_division` | `municipality` | Antwerpen (stad) |
| Statistical sector | `census_areas.area_code` | — | 11002A000 |

---

## Data sources

| Data | Source | License |
|---|---|---|
| Statistical sectors (geometry) | Statbel open data | CC BY 4.0 |
| Population by sector | Statbel open data | CC BY 4.0 |
| Facilities (POI) | Geofabrik / OpenStreetMap | ODbL |
| Road network (routing) | Geofabrik / OpenStreetMap | ODbL |
