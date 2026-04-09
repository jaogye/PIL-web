# PIL – Public Infrastructure Locator

**PIL** is a web-based spatial decision-support tool for optimal placement of public facilities — schools, health centers, hospitals, and similar services. PIL answers the question:

> *"Where should we build the next N facilities to best serve the population?"*

This repository is the **web migration** of the original LIP2 Java desktop prototype. The optimization engine has been ported to Python and the interface rebuilt as a React single-page application, extended with multi-country support.

---

## Features

| Feature | Description |
|---|---|
| **P-Median** | Minimise total demand-weighted travel time — maximises efficiency |
| **P-Center** | Minimise the maximum travel time — maximises equity |
| **Maximum Coverage** | Maximise population served within a time radius (momentum-based alternating greedy + capacitated assignment + redundancy pruning) |
| **Bump Hunter** | Identify high-demand clusters as exploratory facility placement candidates |
| **Capacity Rebalancing** | Redistribute capacity across existing facilities to reduce unmet demand |
| **Reoptimization** | Fix user-selected facility locations and re-optimize the remainder |
| **Excel / JSON Reports** | One-click export of results for planning documents |
| **Interactive Map** | Visualise facility locations on a MapLibre GL map with layer control |
| **Multi-Country** | Switch between country databases (Ecuador, Belgium) via a dropdown |
| **Geographic Scope** | Filter optimization by political divisions (region > province > municipality) |
| **Target Population** | Per-facility-type population groups (school-age, patients, etc.) for demand weighting |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Frontend  │  React 18 + MapLibre GL + React Query   │
│            │  Served by nginx                        │
├──────────────────────────────────────────────────────┤
│  Backend   │  FastAPI (Python 3.12)                  │
│            │  Optimization: NumPy                    │
│            │  ORM: SQLAlchemy 2 (async)               │
├──────────────────────────────────────────────────────┤
│  Database  │  PostgreSQL 16 + PostGIS 3               │
│            │  One database per country                │
├──────────────────────────────────────────────────────┤
│  Routing   │  OSRM (self-hosted via Docker)           │
│            │  Used to build travel-time matrices      │
└──────────────────────────────────────────────────────┘
```

### Multi-Database Design

The backend supports multiple country databases via a `X-LIP2-Database` HTTP header sent by the frontend. Each country database is a fully independent PostgreSQL database sharing the same schema. The frontend dropdown switches the active database, clears the map, and refetches all country-specific data (political divisions tree, facilities, scenarios).

---

## Supported Countries

| Country | Database | Census Source | Routing | Status |
|---|---|---|---|---|
| Ecuador | `lip2_ecuador` | INEC census areas | Pre-computed raster model | Production |
| Belgium | `lip2_belgium` | Statbel statistical sectors 2024 | OSRM (OpenStreetMap) | Production |

---

## Repository Structure

```
PIL-web/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app entry point + /api/v1/databases
│   │   ├── config.py                # Settings (env vars, available_databases list)
│   │   ├── database.py              # Per-country async SQLAlchemy engine pool
│   │   ├── dependencies.py          # X-LIP2-Database header → DB session injection
│   │   ├── api/routes/
│   │   │   ├── optimization.py      # POST /optimization/run · POST /{id}/rebalance
│   │   │   ├── infrastructure.py    # CRUD /infrastructure/
│   │   │   ├── impacts.py           # POST /impacts/calculate
│   │   │   ├── reports.py           # GET  /reports/scenario/{id}/excel|json
│   │   │   ├── political_divisions.py  # GET /political-divisions/tree
│   │   │   └── target_population.py # GET /target-population/
│   │   ├── models/                  # SQLAlchemy ORM models
│   │   ├── schemas/                 # Pydantic request/response schemas
│   │   └── optimization/            # Core algorithms
│   │       ├── sparse_matrix.py     # CSR + CSC sparse distance matrix
│   │       ├── p_median.py          # Greedy add + 1-opt exchange
│   │       ├── p_center.py          # L-Layered search + greedy set cover
│   │       ├── max_coverage.py      # GRASP + CMCLP-CAC capacitated assignment
│   │       ├── rebalancing.py       # Capacity rebalancing heuristic
│   │       └── bump_hunter.py       # Gravity-weighted local-maxima detection
│   ├── scripts/
│   │   └── belgium/                 # Belgium ETL pipeline (see below)
│   ├── Dockerfile
│   ├── fly.toml
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Root layout, reoptimization flow, comparison overlay
│   │   ├── components/
│   │   │   ├── Map/MapView.jsx      # MapLibre GL map, right-click menus, rebalancing lines
│   │   │   └── Optimization/
│   │   │       ├── OptimizationPanel.jsx       # Optimization form, results, rebalancing UI
│   │   │       └── PoliticalDivisionTree.jsx   # Hierarchical scope filter tree
│   │   └── services/api.js          # Axios API client (X-LIP2-Database header)
│   ├── Dockerfile
│   ├── fly.toml
│   └── nginx.conf
│
├── database/
│   └── migrations/
│       ├── 001_initial_schema.sql        # Core schema (census_areas, facilities, etc.)
│       ├── 002_served_areas.sql          # Served-area result storage
│       ├── 003_facility_type_lookup.sql  # Facility type reference table
│       ├── 004_target_population.sql     # Census groups + per-area population
│       ├── 005_avg_speed_kmh.sql         # Median routing speed per census area
│       └── 006_bump_hunter_model_type.sql # Adds bump_hunter to model_type enum
│
└── docker-compose.yml               # Full local stack (db + api + frontend)
```

---

## Quick Start (Local)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Git

### Run the full stack

```bash
git clone https://github.com/YOUR_USERNAME/PIL-web.git
cd PIL-web

# Start all services (PostgreSQL + PostGIS, FastAPI, React/nginx)
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000  |
| API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| API Docs (ReDoc) | http://localhost:8000/redoc |

### Environment variables (docker-compose.yml)

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://lip2:lip2@db:5432/lip2_ecuador` | Primary DB connection |
| `AVAILABLE_DATABASES` | `'["lip2_ecuador","lip2_belgium"]'` | JSON array of enabled databases |
| `ALLOWED_ORIGINS` | `'["http://localhost:5173","http://localhost:3000"]'` | CORS origins |
| `DEBUG` | `true` | Enable debug mode |

### Run backend tests

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

---

## Optimization Algorithms

All algorithms operate on a **sparse travel-time matrix** between census areas and a **demand vector** (population per census area). No external LP solver is required — all models use custom heuristics implemented in NumPy.

### P-Median

**Objective:** minimise `Σ demand[i] × dist(i, nearest facility)`.
**Algorithm:** Greedy-Add phase followed by 1-opt Exchange (fully vectorised, skipped for n > 10,000).
**Use case:** Efficiency — minimises average travel time weighted by population.

### P-Center

**Objective:** minimise `max dist(i, nearest facility)` for all areas i.
**Algorithm:** L-Layered Search (Kramer, Iori, Vidal 2018) with a greedy dominating-set feasibility oracle.
**Use case:** Equity — guarantees no area is farther than a threshold from a facility.

### Maximum Coverage (MCLP)

**Objective:** maximise `Σ demand[i]` for all areas within service radius R.
**Algorithm:** Momentum-based alternating greedy (CMCLP-CAC). A momentum score `momentum[i] = Σ_j remaining[j] · speed(i,j)` drives facility placement in alternating MAX turns (densest uncovered cluster) and MIN turns (most peripheral area). Each placed facility fills demand nearest-first up to `cap_min`. Redundancy pruning (Phase 2B) iteratively removes facilities whose served areas can be absorbed by neighbours without exceeding `cap_max`.
**Use case:** Budget-constrained planning — maximise the number of people served within a time budget.

### Bump Hunter

**Objective:** identify census areas that are local maxima of a gravity-weighted demand score.
**Algorithm:** Gravity score `s[i] = demand[i] + Σ demand[j] / (1 + dist(j → i))` over k-nearest neighbours; local maximum detection via KDTree spatial KNN (fallback to CSR neighbours for n > 2,000).
**Use case:** Exploratory analysis — suggest high-demand clusters without fixing the number of facilities p.

### Capacity Rebalancing

**Objective:** reduce unmet demand by transferring surplus capacity from over-served facilities to under-served ones, without changing facility locations.

**Algorithm:** Greedy transfer heuristic. Each census area is assigned to its nearest facility (no radius restriction). Then, iteratively:
1. Find the facility with the highest unmet demand (deficit = assigned demand − capacity).
2. Find the facility with the highest available surplus (surplus = capacity − assigned demand, above the operational floor).
3. Transfer `min(surplus, deficit)` capacity units from donor to recipient.
4. Repeat until no unmet demand remains or the transfer limit is reached.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `capacity_per_facility` | Average covered demand | Uniform capacity target assigned to all facilities before rebalancing starts |
| `min_capacity` | 0 | Operational floor — the minimum capacity any facility must retain after transfers |
| `max_transfers` | 20 | Maximum number of transfer operations to propose |

**Output:**
- List of recommended transfers: *from facility → to facility*, amount transferred, estimated impact on unmet demand.
- Updated capacity for each facility after all transfers.
- Unmet demand before and after, and percentage improvement.
- Transfers are drawn on the map as orange lines with thickness proportional to the transferred amount. Click a line to see the transfer details.

**Use case:** Improve an existing network without relocating or building facilities — redistribute staff, equipment, or budget to where it is most needed.

---

## Workflow

### 1. Run an Optimization

1. Select a **model** (P-Median, P-Center, or Maximum Coverage).
2. Choose a **facility type** (school, high school, health center, hospital).
3. Set the **number of facilities** p (or service radius for Max Coverage).
4. Optionally apply **capacity constraints** (min/max demand per facility) and a **geographic scope**.
5. Click **Run Optimization**. The job runs in the background; the panel polls every 30 s.
6. Results appear on the map: blue circles for new facilities, yellow squares for served census areas, grey dashed lines showing assignments.

### 2. Reoptimize with Manual Adjustments

After an optimization completes, the user can manually adjust the solution directly on the map:

- **Right-click a facility** → "Remove Facility" (marks it for removal, shown in grey with red border).
- **Right-click a census area** → "Add Facility Here" (marks it as a proposed facility, shown in purple).
- The **Manual Edits** overlay (bottom-right of map) shows the pending changes.
- Click **Reoptimize**: the system treats the user's kept and added facilities as fixed, then re-optimizes the remaining positions using the original model and parameters. A new scenario is created with the prefix `Reopt_`.
- A **Comparison overlay** (top-right of map) shows the key metrics of the previous and reoptimized scenarios side by side, with colour-coded deltas.

### 3. Rebalance Capacity

After any optimization scenario is loaded, a **Capacity Rebalancing** section becomes available in the panel:

- Set the **capacity per facility** (target uniform capacity; defaults to average covered demand).
- Set the **minimum operational floor** (capacity a facility must always retain).
- Set the **maximum number of transfers**.
- Click **Run Rebalancing**. The algorithm completes instantly.
- Results show: unmet demand before and after, improvement percentage, and the list of recommended transfers (facility code → facility code, amount).
- Transfers are drawn on the map as **orange lines** with thickness proportional to the transferred amount. Click a line to see the transfer details.

---

## API Reference

The full interactive API documentation is available at `/docs` (Swagger UI) and `/redoc` after the backend is running.

### Key Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/databases` | List available country databases |
| `POST` | `/api/v1/optimization/run` | Submit a facility location optimization (async) |
| `GET` | `/api/v1/optimization/` | List all scenarios |
| `GET` | `/api/v1/optimization/{id}` | Get scenario with full facility locations |
| `DELETE` | `/api/v1/optimization/{id}` | Delete a scenario |
| `POST` | `/api/v1/optimization/{id}/rebalance` | Run capacity rebalancing on a completed scenario |
| `GET` | `/api/v1/infrastructure/` | List existing facilities |
| `POST` | `/api/v1/infrastructure/` | Register a new facility |
| `POST` | `/api/v1/impacts/calculate` | Compute social coverage impact |
| `GET` | `/api/v1/reports/scenario/{id}/excel` | Download Excel report |
| `GET` | `/api/v1/reports/scenario/{id}/json` | Download JSON export |
| `GET` | `/api/v1/political-divisions/tree` | Full political division hierarchy |
| `POST` | `/api/v1/political-divisions/census-summary` | Census summary for selected parishes |
| `GET` | `/api/v1/target-population/` | List census population groups (school-age, patients, etc.) |
| `GET` | `/health` | Health check |

All endpoints that access country data require the `X-LIP2-Database` header specifying the target database (e.g. `lip2_ecuador`).

### Example: Run a P-Median optimization

```bash
curl -X POST http://localhost:8000/api/v1/optimization/run \
  -H "Content-Type: application/json" \
  -H "X-LIP2-Database: lip2_belgium" \
  -d '{
    "name": "Hospitals – Antwerp Province",
    "model_type": "p_median",
    "facility_type": "hospital",
    "p_facilities": 5,
    "mode": "from_scratch"
  }'
```

### Example: Run rebalancing on a completed scenario

```bash
curl -X POST http://localhost:8000/api/v1/optimization/42/rebalance \
  -H "Content-Type: application/json" \
  -H "X-LIP2-Database: lip2_ecuador" \
  -d '{
    "capacity_per_facility": 5000,
    "min_capacity": 500,
    "max_transfers": 20
  }'
```

### Example: Reoptimize with fixed facility locations

```bash
curl -X POST http://localhost:8000/api/v1/optimization/run \
  -H "Content-Type: application/json" \
  -H "X-LIP2-Database: lip2_ecuador" \
  -d '{
    "name": "Reopt_PMedian_HighSchool",
    "model_type": "p_median",
    "p_facilities": 10,
    "mode": "from_scratch",
    "fixed_census_area_ids": [1042, 1087, 2315]
  }'
```

---

## Adding a New Country

1. Create a new PostgreSQL database using the schema in `database/migrations/001_initial_schema.sql`.
2. Populate `political_division`, `census_areas`, `facilities`, and `distance_matrix` tables with country data.
3. Add the new database name to `AVAILABLE_DATABASES` in `docker-compose.yml`.
4. Add a display label in `backend/app/main.py` → `_DB_LABELS`.
5. If using OSRM for the travel-time matrix, follow `backend/scripts/belgium/README.md` as a reference ETL pipeline.

See `backend/scripts/belgium/` for a complete example ETL pipeline (Statbel census data + Geofabrik OSM facilities + OSRM distance matrix).

---

## Belgium ETL Pipeline

The Belgium database was populated using a three-step ETL pipeline located in `backend/scripts/belgium/`:

| Script | Description |
|---|---|
| `00_schema.sql` | Creates the `lip2_belgium` database and all tables |
| `01_load_census_areas.py` | Downloads Statbel statistical sectors (19,795) and population; inserts political divisions (3 regions, 11 provinces, 581 municipalities) |
| `02_load_facilities.py` | Downloads Geofabrik Belgium OSM POI layer; maps OSM feature classes to facility types; inserts ~7,200 facilities |
| `03_distance_matrix.py` | Builds sparse travel-time matrix via OSRM `/table` API (≤550 nearest neighbours per sector); supports `--resume` after interruption |

**Results:** 595 political divisions · 19,795 census areas · ~11.5 M population · 7,194 facilities · ~9.9 M distance pairs

See `backend/scripts/belgium/README.md` for step-by-step instructions.

---

## Deployment on Fly.io

### 1. Install flyctl

```bash
# macOS / Linux
curl -L https://fly.io/install.sh | sh

# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex
```

### 2. Authenticate

```bash
fly auth login
```

### 3. Create and deploy the PostgreSQL database

```bash
fly postgres create --name pil-db --region mia --vm-size shared-cpu-1x --volume-size 10
fly postgres connect -a pil-db   # verify connection
```

### 4. Deploy the backend

```bash
cd backend
fly launch --name pil-api --region mia --no-deploy
fly secrets set DATABASE_URL="<connection-string-from-step-3>"
fly secrets set AVAILABLE_DATABASES='["lip2_ecuador","lip2_belgium"]'
fly deploy
```

### 5. Deploy the frontend

```bash
cd ../frontend
fly launch --name pil-app --region mia --no-deploy
fly deploy
```

### 6. Configure GitHub Actions (CI/CD)

Add a `FLY_API_TOKEN` secret to your GitHub repository:

```bash
fly tokens create deploy -x 9999h
# Copy the token → GitHub → Settings → Secrets → Actions → New secret
```

From now on, every push to `main` automatically tests and deploys both services.

---

## Background

This repository migrates the original LIP2 Java desktop prototype to a modern web stack while preserving optimization quality. It adds multi-country support (Belgium as a second country alongside Ecuador), a self-hosted OSRM routing pipeline for travel-time matrix computation, and a React web interface replacing the original desktop UI.

---

## License

MIT / Author. See [LICENSE](LICENSE) for details.
