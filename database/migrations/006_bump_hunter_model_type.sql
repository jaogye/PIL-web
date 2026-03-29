-- Add 'bump_hunter' to the model_type PostgreSQL enum.
ALTER TYPE model_type ADD VALUE IF NOT EXISTS 'bump_hunter';
