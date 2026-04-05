"""Shared FastAPI dependencies – database selection and authentication."""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_session_factory

settings = get_settings()

# Human-readable labels for each country database.
_DB_LABELS: dict[str, str] = {
    "lip2_ecuador": "Ecuador",
    "lip2_belgium": "Belgium",
}

# ── Password hashing ───────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


# ── JWT helpers ────────────────────────────────────────────────────────────

def create_access_token(user_id: int, email: str) -> str:
    from datetime import datetime, timedelta, timezone
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=settings.jwt_access_expire_hours),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])


# ── Database selection ─────────────────────────────────────────────────────

def get_db_name(
    x_lip2_database: str | None = Header(default=None),
) -> str:
    """Read the target database from the X-LIP2-Database request header."""
    available = settings.available_databases
    if x_lip2_database is None:
        return available[0]
    if x_lip2_database not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown database '{x_lip2_database}'. Available: {available}",
        )
    return x_lip2_database


async def get_db(
    db_name: str = Depends(get_db_name),
) -> AsyncGenerator[AsyncSession, None]:
    """Yield a session for the database chosen by the client header."""
    factory = get_session_factory(db_name)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_auth_db() -> AsyncGenerator[AsyncSession, None]:
    """Always yield a session for the admin database.
    Users, usage logs, and access control are stored there, independent
    of any country-specific database."""
    factory = get_session_factory(settings.admin_database)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_accessible_databases(user: "User", db: AsyncSession) -> list[dict]:  # type: ignore[name-defined]
    """Return the list of databases the user may access, with display labels.

    Admin users always receive every available database.
    Regular users receive only the databases recorded in user_database_access.
    """
    from app.models.user import UserDatabaseAccess

    if user.is_admin:
        db_names = settings.available_databases
    else:
        result = await db.execute(
            select(UserDatabaseAccess.db_name).where(
                UserDatabaseAccess.user_id == user.id
            )
        )
        accessible = {row[0] for row in result.all()}
        # Preserve the canonical order defined in settings.
        db_names = [d for d in settings.available_databases if d in accessible]

    return [{"db_name": d, "label": _DB_LABELS.get(d, d)} for d in db_names]


# ── Authentication dependencies ────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_auth_db),
):
    """Validate JWT and return the authenticated User ORM object."""
    from app.models.user import User

    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise credentials_exc

    user = await db.get(User, user_id)
    if user is None or not user.is_active:
        raise credentials_exc
    return user


async def require_admin(user=Depends(get_current_user)):
    """Require the authenticated user to have admin privileges."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user
