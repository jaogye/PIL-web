"""ORM model for the political_division table (provincia > canton > parroquia)."""

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class PoliticalDivision(Base):
    __tablename__ = "political_division"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    level: Mapped[str] = mapped_column(String(20), nullable=False)  # provincia|canton|parroquia
    parent_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("political_division.id"), nullable=True
    )

    children: Mapped[list["PoliticalDivision"]] = relationship(
        "PoliticalDivision",
        back_populates="parent",
        foreign_keys=[parent_id],
        order_by="PoliticalDivision.name",
    )
    parent: Mapped["PoliticalDivision | None"] = relationship(
        "PoliticalDivision",
        back_populates="children",
        remote_side="PoliticalDivision.id",
        foreign_keys=[parent_id],
    )

    def __repr__(self) -> str:
        return f"<PoliticalDivision {self.level} code={self.code} name={self.name}>"
