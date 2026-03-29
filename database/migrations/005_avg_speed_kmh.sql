-- Migration 005: Add avg_speed_kmh to census_areas
--
-- Stores the median travel speed (km/h) derived from each area's stored
-- distance_matrix neighbours. Used to estimate travel times to areas that
-- are not stored in distance_matrix during capacity-constrained assignment.
--
-- Run against every database (lip2_ecuador, lip2_belgium, …):
--   psql -U lip2 -d lip2_ecuador -f 005_avg_speed_kmh.sql
--   psql -U lip2 -d lip2_belgium -f 005_avg_speed_kmh.sql

ALTER TABLE census_areas
    ADD COLUMN IF NOT EXISTS avg_speed_kmh FLOAT;
