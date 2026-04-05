"""
LIP2 – Localizador de Infraestructura Pública v2
FastAPI application entry point.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.api.routes import (
    optimization, infrastructure, impacts, reports,
    political_divisions, target_population,
)
from app.api.routes import auth as auth_router
from app.api.routes import admin as admin_router
from app.dependencies import get_accessible_databases, get_auth_db, get_current_user
from sqlalchemy.ext.asyncio import AsyncSession

# Attach a stdout handler so INFO messages from the optimization module are
# visible in Docker logs regardless of uvicorn's root logger configuration.
_debug_handler = logging.StreamHandler()
_debug_handler.setLevel(logging.DEBUG)
_debug_handler.setFormatter(
    logging.Formatter("%(levelname)s: [%(name)s] %(message)s")
)
for _logger_name in ("app.optimization", "app.api.routes.optimization"):
    _lg = logging.getLogger(_logger_name)
    _lg.setLevel(logging.INFO)
    _lg.addHandler(_debug_handler)
    _lg.propagate = False

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "REST API for the LIP2 Public Infrastructure Locator. "
        "Provides facility location optimization (p-median, p-center, max-coverage), "
        "social impact analysis, and report generation for public infrastructure planning."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Usage logging middleware ───────────────────────────────────────────────

# Endpoints that are skipped for logging (health checks, auth, static).
_SKIP_LOG_PREFIXES = ("/health", "/docs", "/redoc", "/openapi", "/api/v1/auth/login")


async def _log_request(
    endpoint: str,
    method: str,
    db_name: str | None,
    user_id: int | None,
    duration_ms: int,
    status_code: int,
) -> None:
    """Fire-and-forget coroutine: persist one usage_log row."""
    try:
        from app.database import get_session_factory
        from app.models.user import UsageLog

        factory = get_session_factory(settings.admin_database)
        async with factory() as session:
            session.add(UsageLog(
                user_id=user_id,
                endpoint=endpoint,
                method=method,
                db_name=db_name,
                duration_ms=duration_ms,
                status_code=status_code,
                created_at=datetime.now(timezone.utc),
            ))
            await session.commit()
    except Exception:
        pass  # never let logging break the response


@app.middleware("http")
async def usage_logging_middleware(request: Request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    duration_ms = int((time.perf_counter() - start) * 1000)

    path = request.url.path
    if not any(path.startswith(p) for p in _SKIP_LOG_PREFIXES):
        # Extract user_id from Bearer token without failing the request.
        user_id  = None
        db_name  = request.headers.get("x-lip2-database")
        auth_hdr = request.headers.get("authorization", "")
        if auth_hdr.startswith("Bearer "):
            try:
                from app.dependencies import decode_access_token
                payload = decode_access_token(auth_hdr[7:])
                user_id = int(payload.get("sub", 0)) or None
            except Exception:
                pass

        asyncio.ensure_future(
            _log_request(path, request.method, db_name, user_id, duration_ms, response.status_code)
        )

    return response


# ── Auth router (public – no auth required) ────────────────────────────────
app.include_router(auth_router.router, prefix="/api/v1")

# ── Admin router (admin auth enforced inside each endpoint) ───────────────
app.include_router(admin_router.router, prefix="/api/v1")

# ── Protected routers (require valid JWT) ──────────────────────────────────
_auth_dep = [Depends(get_current_user)]

app.include_router(optimization.router,        prefix="/api/v1", dependencies=_auth_dep)
app.include_router(infrastructure.router,      prefix="/api/v1", dependencies=_auth_dep)
app.include_router(impacts.router,             prefix="/api/v1", dependencies=_auth_dep)
app.include_router(reports.router,             prefix="/api/v1", dependencies=_auth_dep)
app.include_router(political_divisions.router, prefix="/api/v1", dependencies=_auth_dep)
app.include_router(target_population.router,   prefix="/api/v1", dependencies=_auth_dep)


# ── Health check (public) ──────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health_check():
    return JSONResponse({"status": "ok", "version": settings.app_version})


@app.get("/api/v1/databases", tags=["databases"])
async def list_databases(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_auth_db),
):
    """Return databases accessible to the authenticated user."""
    return await get_accessible_databases(current_user, db)
