
-- LIP2 – Initial Database Schema
-- Requires: PostgreSQL 14+ with PostGIS 3 extension.
--
-- Run with: psql -U lip2 -d lip2 -f 001_initial_schema.sql

-- Enable PostGIS for geospatial support.
CREATE EXTENSION IF NOT EXISTS postgis;

-- ─────────────────────────────────────────────────────────────────────────── --
-- Census Areas (AreaItem in original Java code)
-- ─────────────────────────────────────────────────────────────────────────── --
CREATE TABLE IF NOT EXISTS census_areas (
    id              SERIAL          PRIMARY KEY,
    area_code       VARCHAR(20)     UNIQUE NOT NULL,
    name            VARCHAR(255),

    -- Administrative hierarchy (Ecuador: Provincia > Cantón > Parroquia)
    province_code   VARCHAR(10)     NOT NULL,
    canton_code     VARCHAR(10)     NOT NULL,
    parish_code     VARCHAR(10)     NOT NULL,

    -- Population demand (e.g. number of people, students, patients)
    demand          FLOAT           NOT NULL DEFAULT 0.0,

    -- Current service capacity (used for capacitated variants)
    capacity        FLOAT           NOT NULL DEFAULT 0.0,

    -- Geographic coordinates (WGS84)
    geom            GEOMETRY(Point, 4326),
    x               FLOAT,  -- longitude (fast access without PostGIS)
    y               FLOAT,  -- latitude

    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_census_geom   ON census_areas USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_census_admin  ON census_areas (province_code, canton_code, parish_code);
CREATE INDEX IF NOT EXISTS idx_census_code   ON census_areas (area_code);


-- ─────────────────────────────────────────────────────────────────────────── --
-- Travel-Time Distance Matrix (Distancia.java)
-- ─────────────────────────────────────────────────────────────────────────── --
-- Stores the travel time (in minutes) between pairs of census areas.
-- This is the core input for all optimization algorithms.
-- For large territories, use sparse storage (only store entries ≤ max_radius).
CREATE TABLE IF NOT EXISTS distance_matrix (
    from_area_id        INTEGER     NOT NULL REFERENCES census_areas(id) ON DELETE CASCADE,
    to_area_id          INTEGER     NOT NULL REFERENCES census_areas(id) ON DELETE CASCADE,
    travel_time_minutes FLOAT       NOT NULL,
    PRIMARY KEY (from_area_id, to_area_id)
);

CREATE INDEX IF NOT EXISTS idx_dist_from ON distance_matrix (from_area_id);
CREATE INDEX IF NOT EXISTS idx_dist_to   ON distance_matrix (to_area_id);


-- ─────────────────────────────────────────────────────────────────────────── --
-- Public Infrastructure Facilities (ClusterItem, fInfraActual)
-- ─────────────────────────────────────────────────────────────────────────── --
CREATE TYPE facility_type   AS ENUM ('school','high_school','health_center','hospital','other');
CREATE TYPE facility_status AS ENUM ('existing','proposed','optimized');

CREATE TABLE IF NOT EXISTS facilities (
    id              SERIAL              PRIMARY KEY,
    name            VARCHAR(255),
    facility_type   facility_type       NOT NULL DEFAULT 'other',
    status          facility_status     NOT NULL DEFAULT 'existing',
    capacity        FLOAT               NOT NULL DEFAULT 0.0,
    census_area_id  INTEGER             REFERENCES census_areas(id),
    geom            GEOMETRY(Point, 4326),
    created_at      TIMESTAMPTZ         NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_facility_geom ON facilities USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_facility_type ON facilities (facility_type);


-- ─────────────────────────────────────────────────────────────────────────── --
-- Optimization Scenarios
-- ─────────────────────────────────────────────────────────────────────────── --
CREATE TYPE model_type      AS ENUM ('p_median','p_center','max_coverage');
CREATE TYPE scenario_status AS ENUM ('pending','running','completed','failed');

CREATE TABLE IF NOT EXISTS optimization_scenarios (
    id              SERIAL              PRIMARY KEY,
    name            VARCHAR(255)        NOT NULL,
    description     VARCHAR(1000),
    model_type      model_type          NOT NULL,
    p_facilities    INTEGER             NOT NULL CHECK (p_facilities >= 1),
    service_radius  FLOAT,              -- minutes; required for max_coverage and p_center
    scope_filters   JSONB,              -- {province_codes, canton_codes, parish_codes}
    parameters      JSONB,              -- extra solver parameters
    status          scenario_status     NOT NULL DEFAULT 'pending',
    result_stats    JSONB,              -- summary metrics from the solver
    error_message   VARCHAR(2000),
    created_at      TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_scenario_status ON optimization_scenarios (status);
CREATE INDEX IF NOT EXISTS idx_scenario_model  ON optimization_scenarios (model_type);


-- ─────────────────────────────────────────────────────────────────────────── --
-- Optimization Results  (one row per selected facility location)
-- ─────────────────────────────────────────────────────────────────────────── --
CREATE TABLE IF NOT EXISTS optimization_results (
    id              SERIAL      PRIMARY KEY,
    scenario_id     INTEGER     NOT NULL REFERENCES optimization_scenarios(id) ON DELETE CASCADE,
    census_area_id  INTEGER     NOT NULL REFERENCES census_areas(id),
    covered_demand  FLOAT,
    assigned_areas  INTEGER,
    max_travel_time FLOAT
);

CREATE INDEX IF NOT EXISTS idx_result_scenario ON optimization_results (scenario_id);
CREATE INDEX IF NOT EXISTS idx_result_area     ON optimization_results (census_area_id);


-- ─────────────────────────────────────────────────────────────────────────── --
-- Interventions  (fIntervencion – point, linear, polygon)
-- ─────────────────────────────────────────────────────────────────────────── --
CREATE TYPE intervention_type AS ENUM ('puntual','lineal','poligonal');

CREATE TABLE IF NOT EXISTS interventions (
    id                  SERIAL              PRIMARY KEY,
    name                VARCHAR(255)        NOT NULL,
    intervention_type   intervention_type   NOT NULL DEFAULT 'puntual',
    facility_type       facility_type,
    estimated_cost      FLOAT               NOT NULL DEFAULT 0.0,
    parameters          JSONB,
    geom                GEOMETRY,           -- Point, LineString, or Polygon
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_intervention_geom ON interventions USING GIST (geom);
