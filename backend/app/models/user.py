"""ORM models for users, password reset tokens, and usage logs."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id:              Mapped[int]           = mapped_column(Integer, primary_key=True, autoincrement=True)
    email:           Mapped[str]           = mapped_column(String(255), unique=True, nullable=False)
    full_name:       Mapped[str | None]    = mapped_column(String(255), nullable=True)
    hashed_password: Mapped[str]           = mapped_column(String(255), nullable=False)
    is_admin:        Mapped[bool]          = mapped_column(Boolean, nullable=False, default=False)
    is_active:       Mapped[bool]          = mapped_column(Boolean, nullable=False, default=True)
    must_change_pw:  Mapped[bool]          = mapped_column(Boolean, nullable=False, default=True)
    created_at:      Mapped[datetime]      = mapped_column(DateTime(timezone=True), nullable=False)
    last_login_at:   Mapped[datetime|None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<User email={self.email} admin={self.is_admin}>"


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id:         Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id:    Mapped[int]      = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token:      Mapped[str]      = mapped_column(String(255), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used:       Mapped[bool]     = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id:          Mapped[int]       = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id:     Mapped[int|None]  = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    endpoint:    Mapped[str]       = mapped_column(String(255), nullable=False)
    method:      Mapped[str]       = mapped_column(String(10), nullable=False, default="GET")
    db_name:     Mapped[str|None]  = mapped_column(String(100), nullable=True)
    duration_ms: Mapped[int|None]  = mapped_column(Integer, nullable=True)
    status_code: Mapped[int|None]  = mapped_column(Integer, nullable=True)
    extra:       Mapped[dict|None] = mapped_column(JSONB, nullable=True)
    created_at:  Mapped[datetime]  = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_usage_user",     "user_id"),
        Index("idx_usage_time",     "created_at"),
        Index("idx_usage_endpoint", "endpoint"),
    )


class UserDatabaseAccess(Base):
    """Records which country databases a user is allowed to access."""
    __tablename__ = "user_database_access"

    id:      Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    db_name: Mapped[str] = mapped_column(String(100), nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "db_name", name="uq_user_database_access"),
        Index("idx_uda_user", "user_id"),
    )
