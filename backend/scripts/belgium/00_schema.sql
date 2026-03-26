-- ============================================================
-- LIP2 Belgium – Full Database Schema
-- Run with:
--   docker exec -i lip2-web-db-1 psql -U lip2 -f - < 00_schema.sql
-- ============================================================

-- 1. Create database
SELECT 'CREATE DATABASE lip2_belgium OWNER lip2'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'lip2_belgium')\gexec

\connect lip2_belgium

-- 2. Extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

-- 3. ENUMs
CREATE TYPE facility_status   AS ENUM ('existing','proposed','optimized');
CREATE TYPE intervention_type AS ENUM ('puntual','lineal','poligonal');
CREATE TYPE model_type        AS ENUM ('p_median','p_center','max_coverage');
CREATE TYPE scenario_status   AS ENUM ('pending','running','completed','failed');

-- 4. political_division  (Region > Province > Municipality)
CREATE TABLE political_division (
    id        SERIAL        PRIMARY KEY,
    code      VARCHAR(20)   UNIQUE NOT NULL,
    name      VARCHAR(255),
    level     VARCHAR(20)   NOT NULL
                            CHECK (level IN ('region','province','municipality')),
    parent_id INTEGER       REFERENCES political_division(id),
    geom      GEOMETRY(MultiPolygon, 4326)
);
CREATE INDEX idx_pd_level  ON political_division (level);
CREATE INDEX idx_pd_parent ON political_division (parent_id);

-- 5. census_areas
CREATE TABLE census_areas (
    id              SERIAL          PRIMARY KEY,
    area_code       VARCHAR(20)     UNIQUE NOT NULL,
    name            VARCHAR(255),
    province_code   VARCHAR(10)     NOT NULL,   -- NIS province code (2 chars)
    canton_code     VARCHAR(10)     NOT NULL,   -- NIS arrondissement code (3 chars)
    parish_code     VARCHAR(10)     NOT NULL,   -- NIS municipality code (5 chars)
    demand          FLOAT           NOT NULL DEFAULT 0.0,
    capacity        FLOAT           NOT NULL DEFAULT 0.0,
    geom            GEOMETRY(Point, 4326),
    x               FLOAT,
    y               FLOAT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    parish_id       INTEGER         REFERENCES political_division(id),
    pop_center_x    FLOAT,
    pop_center_y    FLOAT,
    pop_center_lon  FLOAT,
    pop_center_lat  FLOAT,
    locality_count  INTEGER         DEFAULT 0,
    center_method   VARCHAR(20),
    raster_col      INTEGER,
    raster_row      INTEGER,
    raster_travel_time FLOAT
);
CREATE INDEX idx_census_geom  ON census_areas USING GIST (geom);
CREATE INDEX idx_census_admin ON census_areas (province_code, canton_code, parish_code);
CREATE INDEX idx_census_code  ON census_areas (area_code);
CREATE INDEX idx_census_parish ON census_areas (parish_id);

-- 6. distance_matrix (sparse, ≤ 500 nearest neighbours per area)
CREATE TABLE distance_matrix (
    from_area_id        INTEGER NOT NULL REFERENCES census_areas(id) ON DELETE CASCADE,
    to_area_id          INTEGER NOT NULL REFERENCES census_areas(id) ON DELETE CASCADE,
    travel_time_minutes FLOAT   NOT NULL,
    PRIMARY KEY (from_area_id, to_area_id)
);
CREATE INDEX idx_dist_from ON distance_matrix (from_area_id);
CREATE INDEX idx_dist_to   ON distance_matrix (to_area_id);

-- raster_distance_matrix (optional, same layout as Ecuador)
CREATE TABLE raster_distance_matrix (
    from_area_id        INTEGER NOT NULL,
    to_area_id          INTEGER NOT NULL,
    travel_time_minutes FLOAT   NOT NULL
);
CREATE INDEX rdm_from_idx ON raster_distance_matrix (from_area_id);
CREATE INDEX rdm_to_idx   ON raster_distance_matrix (to_area_id);

-- 7. facility_types lookup
CREATE TABLE facility_types (
    code  VARCHAR(50)  PRIMARY KEY,
    label VARCHAR(255) NOT NULL
);
INSERT INTO facility_types (code, label) VALUES
    ('school',        'Basisschool / École primaire'),
    ('high_school',   'Middelbare school / École secondaire'),
    ('health_center', 'Gezondheidscentrum / Centre de santé'),
    ('hospital',      'Ziekenhuis / Hôpital'),
    ('other',         'Andere / Autre');

-- 8. facilities
CREATE TABLE facilities (
    id              SERIAL          PRIMARY KEY,
    name            VARCHAR(255),
    facility_type   VARCHAR(50)     NOT NULL DEFAULT 'other'
                                    REFERENCES facility_types(code),
    status          facility_status NOT NULL DEFAULT 'existing',
    capacity        FLOAT           NOT NULL DEFAULT 0.0,
    census_area_id  INTEGER         REFERENCES census_areas(id),
    geom            GEOMETRY(Point, 4326),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_facility_geom ON facilities USING GIST (geom);
CREATE INDEX idx_facility_type ON facilities (facility_type);

-- 9. optimization_scenarios
CREATE TABLE optimization_scenarios (
    id              SERIAL          PRIMARY KEY,
    name            VARCHAR(255)    NOT NULL,
    description     VARCHAR(1000),
    model_type      model_type      NOT NULL,
    p_facilities    INTEGER         NOT NULL CHECK (p_facilities >= 1),
    service_radius  FLOAT,
    scope_filters   JSONB,
    parameters      JSONB,
    status          scenario_status NOT NULL DEFAULT 'pending',
    result_stats    JSONB,
    error_message   VARCHAR(2000),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);
CREATE INDEX idx_scenario_status ON optimization_scenarios (status);
CREATE INDEX idx_scenario_model  ON optimization_scenarios (model_type);

-- 10. optimization_results
CREATE TABLE optimization_results (
    id              SERIAL  PRIMARY KEY,
    scenario_id     INTEGER NOT NULL REFERENCES optimization_scenarios(id) ON DELETE CASCADE,
    census_area_id  INTEGER NOT NULL REFERENCES census_areas(id),
    covered_demand  FLOAT,
    assigned_areas  INTEGER,
    max_travel_time FLOAT,
    served_area_ids JSONB
);
CREATE INDEX idx_result_scenario ON optimization_results (scenario_id);
CREATE INDEX idx_result_area     ON optimization_results (census_area_id);

-- 11. interventions
CREATE TABLE interventions (
    id                  SERIAL              PRIMARY KEY,
    name                VARCHAR(255)        NOT NULL,
    intervention_type   intervention_type   NOT NULL DEFAULT 'puntual',
    facility_type       VARCHAR(50)         REFERENCES facility_types(code),
    estimated_cost      FLOAT               NOT NULL DEFAULT 0.0,
    parameters          JSONB,
    geom                GEOMETRY,
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_intervention_geom ON interventions USING GIST (geom);
