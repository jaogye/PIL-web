"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_list_delimiter=",",
    )

    # ------------------------------------------------------------------ #
    # Application                                                          #
    # ------------------------------------------------------------------ #
    app_name: str = "LIP2 API"
    app_version: str = "1.0.0"
    debug: bool = False

    # ------------------------------------------------------------------ #
    # Database                                                             #
    # ------------------------------------------------------------------ #
    database_url: str = "postgresql+asyncpg://lip2:lip2@localhost:5432/lip2_ecuador"

    # Databases available for selection. First entry is the default.
    available_databases: list[str] = ["lip2_ecuador", "lip2_belgium"]

    # ------------------------------------------------------------------ #
    # Security                                                             #
    # ------------------------------------------------------------------ #
    secret_key: str = "change-me-in-production"
    allowed_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ------------------------------------------------------------------ #
    # Optimization limits                                                  #
    # ------------------------------------------------------------------ #
    max_areas_per_request: int = 5000
    max_facilities: int = 200


@lru_cache
def get_settings() -> Settings:
    return Settings()
