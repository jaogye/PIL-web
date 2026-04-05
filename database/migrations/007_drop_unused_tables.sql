-- Migration 007: Drop unused tables.
--
-- Removes tables that have no corresponding model, route, or query in the
-- application backend:
--   - interventions          (exists in Ecuador and Belgium schemas)
--   - raster_distance_matrix (exists only in Belgium schema)
--
-- Both DROP statements use IF EXISTS so this migration is safe to run
-- against either database.
--
-- Run against each database:
--   psql -U lip2 -d lip2_ecuador  -f 007_drop_unused_tables.sql
--   psql -U lip2 -d lip2_belgium  -f 007_drop_unused_tables.sql

DROP TABLE IF EXISTS interventions;
DROP TYPE  IF EXISTS intervention_type;

DROP TABLE IF EXISTS raster_distance_matrix;
