-- Migration 010: Enforce unique (scenario_id, census_area_id) in optimization_results
--
-- Each facility location must appear at most once per scenario.
-- Duplicate rows were produced in COMPLETE_EXISTING mode when multiple DB
-- facilities shared the same census_area_id, causing the pre-selected list to
-- carry duplicate indices that were then inserted as separate rows.
--
-- Before adding the constraint, remove any pre-existing duplicates by keeping
-- only the row with the highest id (most recent insert) for each pair.

DELETE FROM optimization_results
WHERE id NOT IN (
    SELECT MAX(id)
    FROM optimization_results
    GROUP BY scenario_id, census_area_id
);

ALTER TABLE optimization_results
    ADD CONSTRAINT uq_optimization_results_scenario_area
    UNIQUE (scenario_id, census_area_id);
