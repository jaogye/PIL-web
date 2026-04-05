-- Migration 008: User authentication and usage logging.
--
-- Run ONLY against the default database (lip2_ecuador):
--   psql -U lip2 -d lip2_ecuador -f 008_users_and_logs.sql
--
-- Users and logs are global (not per-country) and are always stored
-- in the first available database (lip2_ecuador).

-- ── Users ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL       PRIMARY KEY,
    email           VARCHAR(255) UNIQUE NOT NULL,
    full_name       VARCHAR(255),
    hashed_password VARCHAR(255) NOT NULL,
    is_admin        BOOLEAN      NOT NULL DEFAULT FALSE,
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    -- Forces password change on first login (set for new / reset accounts).
    must_change_pw  BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_login_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);

-- ── Password reset tokens ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id          SERIAL       PRIMARY KEY,
    user_id     INTEGER      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token       VARCHAR(255) UNIQUE NOT NULL,
    expires_at  TIMESTAMPTZ  NOT NULL,
    used        BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prt_token   ON password_reset_tokens (token);
CREATE INDEX IF NOT EXISTS idx_prt_user    ON password_reset_tokens (user_id);

-- ── Usage logs ─────────────────────────────────────────────────────────────
-- Stores one row per API request to track resource consumption per user.
CREATE TABLE IF NOT EXISTS usage_logs (
    id          BIGSERIAL    PRIMARY KEY,
    user_id     INTEGER      REFERENCES users(id),   -- NULL = unauthenticated
    endpoint    VARCHAR(255) NOT NULL,
    method      VARCHAR(10)  NOT NULL DEFAULT 'GET',
    db_name     VARCHAR(100),
    duration_ms INTEGER,
    status_code INTEGER,
    -- Optional structured metadata (e.g. model_type, p_facilities).
    extra       JSONB,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_usage_user     ON usage_logs (user_id);
CREATE INDEX IF NOT EXISTS idx_usage_time     ON usage_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_usage_endpoint ON usage_logs (endpoint);
