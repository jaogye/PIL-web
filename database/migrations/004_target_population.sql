-- Migration 004: Target population support.
--
-- Introduces age-segmented demand so each facility type uses its
-- relevant sub-population (e.g. 6-11 years for schools).
--
-- Changes:
--   1. Create target_population  – catalog of demographic groups.
--   2. Create census_areas_population – population per area per group.
--   3. Add default_target_population_id FK to facility_types.
--   4. Insert nursery facility type.
--   5. Migrate existing census_areas.demand → all_ages group.
--   6. Drop census_areas.demand (replaced by census_areas_population).
--
-- Run against EACH country database:
--   psql -U lip2 -d lip2_ecuador  -f 004_target_population.sql
--   psql -U lip2 -d lip2_belgium  -f 004_target_population.sql

-- ── 1. Catalog of demographic target groups ───────────────────────────────────
CREATE TABLE IF NOT EXISTS target_population (
    id      SERIAL       PRIMARY KEY,
    code    VARCHAR(50)  UNIQUE NOT NULL,
    label   VARCHAR(255) NOT NULL,
    min_age INT,          -- NULL = no lower bound
    max_age INT           -- NULL = no upper bound
);

INSERT INTO target_population (code, label, min_age, max_age) VALUES
    ('age_0_3',   'Infants (0–3 years)',   0,    3),
    ('age_6_11',  'Children (6–11 years)', 6,   11),
    ('age_12_17', 'Youth (12–17 years)',  12,   17),
    ('all_ages',  'Total population',    NULL, NULL)
ON CONFLICT (code) DO NOTHING;

-- ── 2. Population per census area per demographic group ───────────────────────
CREATE TABLE IF NOT EXISTS census_areas_population (
    id                   SERIAL PRIMARY KEY,
    census_area_id       INT    NOT NULL REFERENCES census_areas(id) ON DELETE CASCADE,
    target_population_id INT    NOT NULL REFERENCES target_population(id),
    population           FLOAT  NOT NULL DEFAULT 0.0,
    UNIQUE (census_area_id, target_population_id)
);

CREATE INDEX IF NOT EXISTS idx_cap_area ON census_areas_population (census_area_id);
CREATE INDEX IF NOT EXISTS idx_cap_tp   ON census_areas_population (target_population_id);

-- ── 3. Link facility types to their default demographic group ─────────────────
ALTER TABLE facility_types
    ADD COLUMN IF NOT EXISTS default_target_population_id INT
        REFERENCES target_population(id);

-- ── 4. Add nursery facility type ──────────────────────────────────────────────
INSERT INTO facility_types (code, label, default_target_population_id)
VALUES (
    'nursery',
    'Guardería infantil',
    (SELECT id FROM target_population WHERE code = 'age_0_3')
)
ON CONFLICT (code) DO NOTHING;

-- ── 5. Set default target populations for existing facility types ─────────────
UPDATE facility_types
SET default_target_population_id = (SELECT id FROM target_population WHERE code = 'age_6_11')
WHERE code = 'school';

UPDATE facility_types
SET default_target_population_id = (SELECT id FROM target_population WHERE code = 'age_12_17')
WHERE code = 'high_school';

UPDATE facility_types
SET default_target_population_id = (SELECT id FROM target_population WHERE code = 'all_ages')
WHERE code IN ('health_center', 'hospital', 'other');

-- ── 6. Migrate existing demand values → all_ages group ────────────────────────
-- Preserves the current total-population figures before the column is dropped.
INSERT INTO census_areas_population (census_area_id, target_population_id, population)
SELECT
    ca.id,
    tp.id,
    ca.demand
FROM census_areas ca
CROSS JOIN target_population tp
WHERE tp.code = 'all_ages'
  AND ca.demand IS NOT NULL
ON CONFLICT (census_area_id, target_population_id) DO NOTHING;

-- ── 7. Drop census_areas.demand (now stored in census_areas_population) ───────
ALTER TABLE census_areas DROP COLUMN IF EXISTS demand;
