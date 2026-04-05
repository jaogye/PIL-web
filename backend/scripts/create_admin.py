"""
Create the initial admin user in the admin database.

Usage:
    python scripts/create_admin.py --email admin@lip2.local --password MySecret123
"""

import argparse
import asyncio
import os
from datetime import datetime, timezone

import bcrypt
import asyncpg

# All country databases to which the admin user will have access.
AVAILABLE_DATABASES: list[str] = ["lip2_ecuador", "lip2_belgium"]


async def main(email: str, password: str, full_name: str) -> None:
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    db_url = os.getenv("DATABASE_URL", "postgresql://lip2:lip2@localhost:5432/admin")
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    # Replace the database name at the end with 'admin' if a different default was set.
    base_url = db_url.rsplit("/", 1)[0]
    admin_url = f"{base_url}/admin"

    conn = await asyncpg.connect(admin_url)
    try:
        existing = await conn.fetchval("SELECT id FROM users WHERE email = $1", email)
        if existing:
            print(f"User {email} already exists (id={existing}).")
            return

        uid = await conn.fetchval(
            """
            INSERT INTO users (email, full_name, hashed_password, is_admin, is_active,
                               must_change_pw, created_at)
            VALUES ($1, $2, $3, TRUE, TRUE, FALSE, $4)
            RETURNING id
            """,
            email, full_name, hashed_pw, datetime.now(timezone.utc),
        )
        print(f"Admin user created  id={uid}  email={email}")

        # Grant access to all available databases so the admin can use them immediately.
        for db_name in AVAILABLE_DATABASES:
            await conn.execute(
                "INSERT INTO user_database_access (user_id, db_name) "
                "VALUES ($1, $2) ON CONFLICT DO NOTHING",
                uid, db_name,
            )
        print(f"Granted access to: {AVAILABLE_DATABASES}")
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email",     default="admin@lip2.local")
    parser.add_argument("--password",  required=True)
    parser.add_argument("--full-name", default="Administrator")
    args = parser.parse_args()
    asyncio.run(main(args.email, args.password, args.full_name))
