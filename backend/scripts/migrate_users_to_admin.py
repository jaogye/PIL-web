"""
Migrate users, password_reset_tokens, and usage_logs from lip2_ecuador
to the new admin database.

Run AFTER applying 009_admin_database.sql to the admin database.

Usage:
    python scripts/migrate_users_to_admin.py [--grant-all]

Options:
    --source-url URL   Source DB (default: postgresql://lip2:lip2@localhost:5432/lip2_ecuador)
    --admin-url  URL   Admin  DB (default: postgresql://lip2:lip2@localhost:5432/admin)
    --grant-all        Grant every migrated user access to all available databases.
                       Use this flag on first migration to preserve existing behaviour.
"""

import argparse
import asyncio
import os

import asyncpg

# Databases that will be offered for selection in the application.
AVAILABLE_DATABASES: list[str] = ["lip2_ecuador", "lip2_belgium"]


async def main(source_url: str, admin_url: str, grant_all: bool) -> None:
    source = await asyncpg.connect(source_url)
    admin  = await asyncpg.connect(admin_url)

    try:
        # ── 1. Migrate users ─────────────────────────────────────────────────
        users = await source.fetch(
            "SELECT id, email, full_name, hashed_password, is_admin, is_active, "
            "       must_change_pw, created_at, last_login_at "
            "FROM users"
        )
        print(f"Found {len(users)} user(s) in source database.")

        # old_id -> new_id mapping needed for foreign keys
        id_map: dict[int, int] = {}

        for u in users:
            existing_id = await admin.fetchval(
                "SELECT id FROM users WHERE email = $1", u["email"]
            )
            if existing_id:
                print(f"  SKIP  {u['email']} — already exists (id={existing_id})")
                id_map[u["id"]] = existing_id
                continue

            new_id = await admin.fetchval(
                """
                INSERT INTO users (email, full_name, hashed_password, is_admin,
                                   is_active, must_change_pw, created_at, last_login_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
                """,
                u["email"], u["full_name"], u["hashed_password"],
                u["is_admin"], u["is_active"], u["must_change_pw"],
                u["created_at"], u["last_login_at"],
            )
            print(f"  OK    {u['email']}  old_id={u['id']} -> new_id={new_id}")
            id_map[u["id"]] = new_id

        # ── 2. Migrate password reset tokens ─────────────────────────────────
        tokens = await source.fetch(
            "SELECT user_id, token, expires_at, used, created_at "
            "FROM password_reset_tokens"
        )
        print(f"\nFound {len(tokens)} password reset token(s) to migrate.")
        migrated_tokens = 0
        for t in tokens:
            new_user_id = id_map.get(t["user_id"])
            if new_user_id is None:
                print(f"  SKIP  token for unknown source user_id={t['user_id']}")
                continue
            already = await admin.fetchval(
                "SELECT id FROM password_reset_tokens WHERE token = $1", t["token"]
            )
            if already:
                continue
            await admin.execute(
                """
                INSERT INTO password_reset_tokens (user_id, token, expires_at, used, created_at)
                VALUES ($1, $2, $3, $4, $5)
                """,
                new_user_id, t["token"], t["expires_at"], t["used"], t["created_at"],
            )
            migrated_tokens += 1
        print(f"  Migrated {migrated_tokens} token(s).")

        # ── 3. Migrate usage logs ─────────────────────────────────────────────
        logs = await source.fetch(
            "SELECT user_id, endpoint, method, db_name, duration_ms, "
            "       status_code, extra, created_at "
            "FROM usage_logs "
            "ORDER BY id"
        )
        print(f"\nFound {len(logs)} usage log row(s) to migrate.")
        if logs:
            rows = [
                (
                    id_map.get(log["user_id"]) if log["user_id"] else None,
                    log["endpoint"],
                    log["method"],
                    log["db_name"],
                    log["duration_ms"],
                    log["status_code"],
                    log["extra"],
                    log["created_at"],
                )
                for log in logs
            ]
            await admin.executemany(
                """
                INSERT INTO usage_logs
                    (user_id, endpoint, method, db_name, duration_ms,
                     status_code, extra, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                rows,
            )
            print(f"  Migrated {len(rows)} log row(s).")

        # ── 4. Grant database access ──────────────────────────────────────────
        if grant_all:
            print(f"\nGranting access to {AVAILABLE_DATABASES} for all users …")
            all_users = await admin.fetch("SELECT id, email FROM users")
            for u in all_users:
                for db_name in AVAILABLE_DATABASES:
                    await admin.execute(
                        "INSERT INTO user_database_access (user_id, db_name) "
                        "VALUES ($1, $2) ON CONFLICT DO NOTHING",
                        u["id"], db_name,
                    )
            print(f"  Access granted for {len(all_users)} user(s).")

        print("\nMigration complete.")

    finally:
        await source.close()
        await admin.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-url",
        default=os.getenv(
            "SOURCE_DB_URL", "postgresql://lip2:lip2@localhost:5432/lip2_ecuador"
        ),
        help="Connection URL for the source database (lip2_ecuador)",
    )
    parser.add_argument(
        "--admin-url",
        default=os.getenv(
            "ADMIN_DB_URL", "postgresql://lip2:lip2@localhost:5432/admin"
        ),
        help="Connection URL for the new admin database",
    )
    parser.add_argument(
        "--grant-all",
        action="store_true",
        help="Grant all migrated users access to every available database",
    )
    args = parser.parse_args()
    asyncio.run(main(args.source_url, args.admin_url, args.grant_all))
