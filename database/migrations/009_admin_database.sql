-- Migration 009: Admin database schema.
--
-- Creates the dedicated 'admin' database that holds all user authentication
-- data, usage logs, and per-user database access control — independent of
-- any country-specific database (lip2_ecuador, lip2_belgium, etc.).
--
-- Step 1: create the database from psql as a superuser:
--   CREATE DATABASE admin OWNER lip2;
--
-- Step 2: apply this schema:
--   psql -U lip2 -d admin -f 009_admin_database.sql
--
-- Step 3: migrate existing data from lip2_ecuador:
--   python backend/scripts/migrate_users_to_admin.py --grant-all

-- ── Users ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL       PRIMARY KEY,
    email           VARCHAR(255) UNIQUE NOT NULL,
    full_name       VARCHAR(255),
    hashed_password VARCHAR(255) NOT NULL,
    is_admin        BOOLEAN      NOT NULL DEFAULT FALSE,
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    must_change_pw  BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_login_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);

-- ── Password reset tokens ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id          SERIAL       PRIMARY KEY,
    user_id     INTEGER      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token       VARCHAR(255) UNIQUE NOT NULL,
    expires_at  TIMESTAMPTZ  NOT NULL,
    used        BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prt_token ON password_reset_tokens (token);
CREATE INDEX IF NOT EXISTS idx_prt_user  ON password_reset_tokens (user_id);

-- ── Usage logs ───────────────────────────────────────────────────────────────
-- One row per API request; db_name records which country DB was targeted.
CREATE TABLE IF NOT EXISTS usage_logs (
    id          BIGSERIAL    PRIMARY KEY,
    user_id     INTEGER      REFERENCES users(id),   -- NULL = unauthenticated
    endpoint    VARCHAR(255) NOT NULL,
    method      VARCHAR(10)  NOT NULL DEFAULT 'GET',
    db_name     VARCHAR(100),
    duration_ms INTEGER,
    status_code INTEGER,
    extra       JSONB,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_usage_user     ON usage_logs (user_id);
CREATE INDEX IF NOT EXISTS idx_usage_time     ON usage_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_usage_endpoint ON usage_logs (endpoint);

-- ── User database access ─────────────────────────────────────────────────────
-- Controls which country databases a regular user can access.
-- Admin users (is_admin = true) bypass this table and always have full access.
CREATE TABLE IF NOT EXISTS user_database_access (
    id       SERIAL       PRIMARY KEY,
    user_id  INTEGER      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    db_name  VARCHAR(100) NOT NULL,
    UNIQUE (user_id, db_name)
);

CREATE INDEX IF NOT EXISTS idx_uda_user ON user_database_access (user_id);
