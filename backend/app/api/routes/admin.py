"""
Admin-only endpoints.

GET    /admin/users            – List all users (with their database access).
POST   /admin/users            – Create a new user and assign database access.
PUT    /admin/users/{id}       – Update user (name, active, admin, password, db access).
POST   /admin/users/{id}/reset – Generate a password-reset link.
GET    /admin/stats            – Aggregated usage statistics.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.dependencies import (
    _DB_LABELS,
    get_accessible_databases,
    get_auth_db,
    hash_password,
    require_admin,
)
from app.models.user import PasswordResetToken, User, UsageLog, UserDatabaseAccess

router   = APIRouter(prefix="/admin", tags=["admin"])
settings = get_settings()


# ── Schemas ────────────────────────────────────────────────────────────────

class UserOut(BaseModel):
    id:                   int
    email:                str
    full_name:            str | None
    is_admin:             bool
    is_active:            bool
    must_change_pw:       bool
    created_at:           str
    last_login_at:        str | None
    # Databases this user can access: [{db_name, label}]
    accessible_databases: list[dict]


class CreateUserRequest(BaseModel):
    email:     str = Field(..., min_length=3)
    full_name: str | None = None
    password:  str = Field(..., min_length=8)
    is_admin:  bool = False
    # Database names the new user is allowed to access.
    # Ignored when is_admin=True (admins always have full access).
    database_access: list[str] = []


class UpdateUserRequest(BaseModel):
    full_name:    str | None = None
    is_admin:     bool | None = None
    is_active:    bool | None = None
    new_password: str | None = Field(None, min_length=8)
    # New list of accessible database names. None = leave unchanged.
    # Pass an empty list to revoke all access.
    database_access: list[str] | None = None


class ResetLinkResponse(BaseModel):
    reset_url:  str
    expires_at: str


class StatsResponse(BaseModel):
    overview:       dict
    user_activity:  list[dict]
    endpoint_usage: list[dict]
    daily_trend:    list[dict]


# ── Helpers ────────────────────────────────────────────────────────────────

def _validate_db_names(db_names: list[str]) -> None:
    """Raise 400 if any db_name is not in available_databases."""
    for db in db_names:
        if db not in settings.available_databases:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown database '{db}'. Available: {settings.available_databases}",
            )


def _format_user(u: User, accessible_dbs: list[dict]) -> UserOut:
    return UserOut(
        id=u.id,
        email=u.email,
        full_name=u.full_name,
        is_admin=u.is_admin,
        is_active=u.is_active,
        must_change_pw=u.must_change_pw,
        created_at=u.created_at.isoformat(),
        last_login_at=u.last_login_at.isoformat() if u.last_login_at else None,
        accessible_databases=accessible_dbs,
    )


async def _load_all_access(db: AsyncSession) -> dict[int, list[dict]]:
    """Return a mapping user_id -> [{db_name, label}] for all users at once."""
    rows = await db.execute(
        select(UserDatabaseAccess.user_id, UserDatabaseAccess.db_name)
    )
    access: dict[int, list[dict]] = {}
    for user_id, db_name in rows.all():
        access.setdefault(user_id, []).append(
            {"db_name": db_name, "label": _DB_LABELS.get(db_name, db_name)}
        )
    # Preserve canonical order for each user.
    canonical = settings.available_databases
    for uid in access:
        access[uid].sort(key=lambda x: canonical.index(x["db_name"])
                         if x["db_name"] in canonical else 999)
    return access


async def _set_user_access(
    db: AsyncSession, user_id: int, db_names: list[str]
) -> None:
    """Replace the user's database access records with db_names."""
    await db.execute(
        delete(UserDatabaseAccess).where(UserDatabaseAccess.user_id == user_id)
    )
    for db_name in db_names:
        db.add(UserDatabaseAccess(user_id=user_id, db_name=db_name))


# ── User management ────────────────────────────────────────────────────────

@router.get("/users", response_model=list[UserOut])
async def list_users(
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_auth_db),
):
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    users = result.scalars().all()

    access_map = await _load_all_access(db)
    all_dbs = [{"db_name": d, "label": _DB_LABELS.get(d, d)} for d in settings.available_databases]

    return [
        _format_user(
            u,
            # Admins always see every database regardless of access records.
            all_dbs if u.is_admin else access_map.get(u.id, []),
        )
        for u in users
    ]


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: CreateUserRequest,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_auth_db),
):
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    _validate_db_names(payload.database_access)

    user = User(
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
        is_admin=payload.is_admin,
        is_active=True,
        must_change_pw=True,
        created_at=datetime.now(timezone.utc),
    )
    db.add(user)
    await db.flush()  # populate user.id before inserting access records

    # Non-admin users get only the explicitly granted databases.
    # Admin users bypass the access table, but we still store their grants
    # so that downgrading them later retains sensible defaults.
    await _set_user_access(db, user.id, payload.database_access)

    await db.commit()
    await db.refresh(user)

    accessible_dbs = await get_accessible_databases(user, db)
    return _format_user(user, accessible_dbs)


@router.put("/users/{user_id}", response_model=UserOut)
async def update_user(
    user_id: int,
    payload: UpdateUserRequest,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_auth_db),
):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if payload.full_name is not None:
        user.full_name = payload.full_name
    if payload.is_admin is not None:
        user.is_admin = payload.is_admin
    if payload.is_active is not None:
        user.is_active = payload.is_active
    if payload.new_password is not None:
        user.hashed_password = hash_password(payload.new_password)
        user.must_change_pw  = True

    if payload.database_access is not None:
        _validate_db_names(payload.database_access)
        await _set_user_access(db, user_id, payload.database_access)

    await db.commit()

    accessible_dbs = await get_accessible_databases(user, db)
    return _format_user(user, accessible_dbs)


@router.post("/users/{user_id}/reset", response_model=ResetLinkResponse)
async def generate_reset_link(
    user_id: int,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_auth_db),
):
    """Generate a one-time password-reset link valid for 24 hours."""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Invalidate previous unused tokens for this user.
    old_tokens = await db.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.user_id == user_id,
            PasswordResetToken.used == False,  # noqa: E712
        )
    )
    for t in old_tokens.scalars().all():
        t.used = True

    token      = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=settings.jwt_reset_token_expire_hours)

    db.add(PasswordResetToken(
        user_id=user_id,
        token=token,
        expires_at=expires_at,
        used=False,
        created_at=datetime.now(timezone.utc),
    ))
    await db.commit()

    reset_url = f"{settings.frontend_url}?reset_token={token}"
    return ResetLinkResponse(reset_url=reset_url, expires_at=expires_at.isoformat())


# ── Usage statistics ───────────────────────────────────────────────────────

@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_auth_db),
):
    now = datetime.now(timezone.utc)

    # ── Overview ─────────────────────────────────────────────────────────
    total_users  = (await db.execute(select(func.count()).select_from(User))).scalar()
    active_users = (await db.execute(
        select(func.count()).select_from(User).where(User.is_active == True)  # noqa: E712
    )).scalar()

    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start  = now - timedelta(days=7)

    req_today = (await db.execute(
        select(func.count()).select_from(UsageLog)
        .where(UsageLog.created_at >= today_start)
    )).scalar()

    req_week = (await db.execute(
        select(func.count()).select_from(UsageLog)
        .where(UsageLog.created_at >= week_start)
    )).scalar()

    req_total = (await db.execute(select(func.count()).select_from(UsageLog))).scalar()

    avg_duration = (await db.execute(
        select(func.avg(UsageLog.duration_ms)).select_from(UsageLog)
        .where(UsageLog.created_at >= week_start, UsageLog.duration_ms.isnot(None))
    )).scalar()

    overview = {
        "total_users":          total_users,
        "active_users":         active_users,
        "requests_today":       req_today,
        "requests_this_week":   req_week,
        "requests_total":       req_total,
        "avg_duration_ms_week": round(avg_duration or 0, 1),
    }

    # ── Per-user activity (last 30 days) ─────────────────────────────────
    month_start = now - timedelta(days=30)
    rows = await db.execute(text("""
        SELECT
            u.id,
            u.email,
            u.full_name,
            u.is_active,
            COUNT(l.id)        AS total_requests,
            AVG(l.duration_ms) AS avg_duration_ms,
            MAX(l.created_at)  AS last_active
        FROM users u
        LEFT JOIN usage_logs l
               ON l.user_id = u.id AND l.created_at >= :since
        GROUP BY u.id, u.email, u.full_name, u.is_active
        ORDER BY total_requests DESC
    """), {"since": month_start})

    user_activity = [
        {
            "user_id":         r.id,
            "email":           r.email,
            "full_name":       r.full_name or "",
            "is_active":       r.is_active,
            "total_requests":  int(r.total_requests or 0),
            "avg_duration_ms": round(float(r.avg_duration_ms or 0), 1),
            "last_active":     r.last_active.isoformat() if r.last_active else None,
        }
        for r in rows
    ]

    # ── Top endpoints (last 30 days) ──────────────────────────────────────
    ep_rows = await db.execute(text("""
        SELECT
            endpoint,
            method,
            COUNT(*)         AS total,
            AVG(duration_ms) AS avg_ms,
            MAX(duration_ms) AS max_ms
        FROM usage_logs
        WHERE created_at >= :since
        GROUP BY endpoint, method
        ORDER BY total DESC
        LIMIT 15
    """), {"since": month_start})

    endpoint_usage = [
        {
            "endpoint": r.endpoint,
            "method":   r.method,
            "total":    int(r.total),
            "avg_ms":   round(float(r.avg_ms or 0), 1),
            "max_ms":   int(r.max_ms or 0),
        }
        for r in ep_rows
    ]

    # ── Daily request trend (last 14 days) ────────────────────────────────
    two_weeks = now - timedelta(days=14)
    day_rows = await db.execute(text("""
        SELECT
            DATE(created_at AT TIME ZONE 'UTC') AS day,
            COUNT(*)                            AS requests,
            COUNT(DISTINCT user_id)             AS unique_users
        FROM usage_logs
        WHERE created_at >= :since
        GROUP BY day
        ORDER BY day
    """), {"since": two_weeks})

    daily_trend = [
        {
            "day":          str(r.day),
            "requests":     int(r.requests),
            "unique_users": int(r.unique_users),
        }
        for r in day_rows
    ]

    return StatsResponse(
        overview=overview,
        user_activity=user_activity,
        endpoint_usage=endpoint_usage,
        daily_trend=daily_trend,
    )
