"""
LIP2 – Localizador de Infraestructura Pública v2
FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.api.routes import optimization, infrastructure, impacts, reports, political_divisions
from fastapi.responses import JSONResponse

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown events."""
    # Future: initialise background task queues, caches, etc.
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

# ------------------------------------------------------------------ #
# CORS                                                                 #
# ------------------------------------------------------------------ #
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
# Routers                                                              #
# ------------------------------------------------------------------ #
app.include_router(optimization.router, prefix="/api/v1")
app.include_router(infrastructure.router, prefix="/api/v1")
app.include_router(impacts.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(political_divisions.router, prefix="/api/v1")


# ------------------------------------------------------------------ #
# Health check                                                         #
# ------------------------------------------------------------------ #
@app.get("/health", tags=["health"])
async def health_check():
    return JSONResponse({"status": "ok", "version": settings.app_version})


# Country labels shown in the frontend dropdown.
_DB_LABELS: dict[str, str] = {
    "lip2_ecuador": "Ecuador",
    "lip2_belgium": "Belgium",
}


@app.get("/api/v1/databases", tags=["databases"])
async def list_databases():
    """Return the list of available databases with their display labels."""
    return [
        {"db_name": db, "label": _DB_LABELS.get(db, db)}
        for db in settings.available_databases
    ]
