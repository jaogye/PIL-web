"""Async SQLAlchemy engine and session factory – multi-database support."""

from __future__ import annotations

import re
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

# ── Derive the base URL (strip the database name at the end) ──────────────
# e.g. "postgresql+asyncpg://lip2:lip2@db:5432/lip2_ecuador"
#    →  "postgresql+asyncpg://lip2:lip2@db:5432/"
_BASE_URL: str = re.sub(r"/[^/]+$", "/", settings.database_url)

# ── Per-database session factory cache ───────────────────────────────────
_sessions: dict[str, async_sessionmaker[AsyncSession]] = {}


def get_session_factory(db_name: str) -> async_sessionmaker[AsyncSession]:
    """Return (creating if necessary) the session factory for *db_name*."""
    if db_name not in _sessions:
        engine = create_async_engine(
            _BASE_URL + db_name,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        _sessions[db_name] = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _sessions[db_name]


# ── Default session factory (backward compat for background tasks) ────────
_default_db: str = settings.database_url.rsplit("/", 1)[-1]

# Keep AsyncSessionLocal pointing at the *first* available database so that
# background tasks can call get_session_factory(db_name) explicitly.
AsyncSessionLocal = get_session_factory(_default_db)

# ── Legacy single engine (kept for compat, points at default DB) ──────────
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency – yields a session for the *default* database.
    Routes that support multi-DB use app.dependencies.get_db instead."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
