-- Migration 003: Replace facility_type ENUM with a lookup table.
--
-- The facility_type ENUM is used in two tables: facilities and interventions.
-- Both columns are migrated to VARCHAR(50) referencing facility_types.code.
--
-- Run with: psql -U lip2 -d lip2 -f 003_facility_type_lookup.sql

-- 1. Create the lookup table.
CREATE TABLE IF NOT EXISTS facility_types (
    code    VARCHAR(50)  PRIMARY KEY,
    label   VARCHAR(255) NOT NULL
);

INSERT INTO facility_types (code, label) VALUES
    ('school',        'Escuela'),
    ('high_school',   'Colegio'),
    ('health_center', 'Centro de Salud'),
    ('hospital',      'Hospital'),
    ('other',         'Otro')
ON CONFLICT (code) DO NOTHING;

-- 2. Migrate facilities.facility_type: ENUM → VARCHAR + FK.
ALTER TABLE facilities
    ALTER COLUMN facility_type TYPE VARCHAR(50) USING facility_type::TEXT;

ALTER TABLE facilities
    ADD CONSTRAINT fk_facilities_facility_type
    FOREIGN KEY (facility_type) REFERENCES facility_types(code);

-- 3. Migrate interventions.facility_type: ENUM → VARCHAR + FK.
--    The column is nullable, so cast NULL safely.
ALTER TABLE interventions
    ALTER COLUMN facility_type TYPE VARCHAR(50) USING facility_type::TEXT;

ALTER TABLE interventions
    ADD CONSTRAINT fk_interventions_facility_type
    FOREIGN KEY (facility_type) REFERENCES facility_types(code);

-- 4. Drop the old ENUM type (no longer referenced).
DROP TYPE IF EXISTS facility_type;
