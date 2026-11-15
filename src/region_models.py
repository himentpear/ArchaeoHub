"""Source-coordinate crops associated with stable artifact identities."""
from sqlalchemy import ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models import Base, _new_uuid


class ArtifactImageRegion(Base):
    __tablename__ = "artifact_image_regions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    artifact_id: Mapped[str] = mapped_column(ForeignKey("artifacts.id", ondelete="CASCADE"))
    canvas_id: Mapped[str] = mapped_column(ForeignKey("canvases.id", ondelete="CASCADE"))
    xywh: Mapped[list] = mapped_column(JSON)
    image_hash: Mapped[str] = mapped_column(String(64))
    source_width: Mapped[int] = mapped_column(Integer)
    source_height: Mapped[int] = mapped_column(Integer)
    view_label: Mapped[str] = mapped_column(String(128), default="")
    media_type: Mapped[str] = mapped_column(String(32), default="line_drawing")
    status: Mapped[str] = mapped_column(String(32), default="approved")
    revision_no: Mapped[int] = mapped_column(Integer, default=1)
