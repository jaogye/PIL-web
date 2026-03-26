"""Shared FastAPI dependencies – database selection via request header."""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_session_factory

settings = get_settings()


def get_db_name(
    x_lip2_database: str | None = Header(default=None),
) -> str:
    """
    Read the target database from the *X-LIP2-Database* request header.
    Falls back to the first entry of settings.available_databases.
    Raises 400 if the value is not in the configured list.
    """
    available = settings.available_databases
    if x_lip2_database is None:
        return available[0]
    if x_lip2_database not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown database '{x_lip2_database}'. "
                   f"Available: {available}",
        )
    return x_lip2_database


async def get_db(
    db_name: str = Depends(get_db_name),
) -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency – yields a session for the database chosen by the client."""
    factory = get_session_factory(db_name)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
