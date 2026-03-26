-- Migration 002: Add served_area_ids column to optimization_results
-- Run with: psql -U lip2 -d lip2 -f 002_served_areas.sql

ALTER TABLE optimization_results
    ADD COLUMN IF NOT EXISTS served_area_ids JSONB;
