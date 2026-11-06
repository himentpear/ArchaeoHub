"""Additive graphic records. Physical OCR and reviewed footnotes are immutable inputs."""
from sqlalchemy import Boolean, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from src.db.models import Base, _new_uuid


class GraphicRun(Base):
    __tablename__ = "graphic_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    manifest_id: Mapped[str] = mapped_column(ForeignKey("manifests.id", ondelete="CASCADE"))
    fingerprint: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="staging")
    published: Mapped[bool] = mapped_column(Boolean, default=False)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)


class GraphicObject(Base):
    __tablename__ = "graphic_objects"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("graphic_runs.id", ondelete="CASCADE"))
    manifest_id: Mapped[str] = mapped_column(ForeignKey("manifests.id", ondelete="CASCADE"))
    category: Mapped[str] = mapped_column(String(32))
    label: Mapped[str] = mapped_column(String(128), default="")
    title: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="candidate")
    revision_no: Mapped[int] = mapped_column(Integer, default=1)
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    structure: Mapped[dict] = mapped_column(JSON, default=dict)


class GraphicFragment(Base):
    __tablename__ = "graphic_fragments"
    __table_args__ = (UniqueConstraint("object_id", "canvas_id", "source_key"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    object_id: Mapped[str] = mapped_column(ForeignKey("graphic_objects.id", ondelete="CASCADE"))
    canvas_id: Mapped[str] = mapped_column(ForeignKey("canvases.id", ondelete="CASCADE"))
    source_key: Mapped[str] = mapped_column(String(128))
    part_index: Mapped[int] = mapped_column(Integer, default=0)
    bbox: Mapped[list] = mapped_column(JSON)
    polygon: Mapped[list] = mapped_column(JSON, default=list)
    rotation: Mapped[int] = mapped_column(Integer, default=0)
    transform: Mapped[dict] = mapped_column(JSON, default=dict)
    source_hash: Mapped[str] = mapped_column(String(64))
    crop_path: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0)
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)


class GraphicCell(Base):
    __tablename__ = "graphic_cells"
    __table_args__ = (UniqueConstraint("fragment_id", "row_index", "column_index"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    fragment_id: Mapped[str] = mapped_column(ForeignKey("graphic_fragments.id", ondelete="CASCADE"))
    row_index: Mapped[int] = mapped_column(Integer)
    column_index: Mapped[int] = mapped_column(Integer)
    row_span: Mapped[int] = mapped_column(Integer, default=1)
    column_span: Mapped[int] = mapped_column(Integer, default=1)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    corrected_text: Mapped[str | None] = mapped_column(Text)
    value_kind: Mapped[str] = mapped_column(String(32), default="unreadable")
    symbols: Mapped[list] = mapped_column(JSON, default=list)
    header_path: Mapped[list] = mapped_column(JSON, default=list)
    anchors: Mapped[list] = mapped_column(JSON, default=list)
    crop_path: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0)
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
