"""Additive storage for local-model TOC proposals; rule ranges stay intact."""
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models import Base, _new_uuid, _utc_now


class TocStructureRun(Base):
    __tablename__ = "toc_structure_runs"
    __table_args__ = (Index("idx_toc_run_manifest", "manifest_id", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    manifest_id: Mapped[str] = mapped_column(String(36), ForeignKey("manifests.id", ondelete="CASCADE"))
    request_key: Mapped[str] = mapped_column(String(64), unique=True)
    source_digest: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(255))
    model_identity: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="running")
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class TocNodeProposal(Base):
    __tablename__ = "toc_node_proposals"
    __table_args__ = (
        UniqueConstraint("run_id", "node_key", name="uq_toc_proposal_run_node"),
        Index("idx_toc_proposal_run", "run_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("toc_structure_runs.id", ondelete="CASCADE"))
    node_key: Mapped[str] = mapped_column(String(32))
    source_range_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("iiif_ranges.id", ondelete="SET NULL"), nullable=True)
    block_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("alto_blocks.id", ondelete="SET NULL"), nullable=True)
    keep: Mapped[bool] = mapped_column(Boolean)
    level: Mapped[int] = mapped_column(Integer)
    parent_key: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    reason: Mapped[str] = mapped_column(Text)
    evidence_quote: Mapped[str] = mapped_column(Text)
    source_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    review_status: Mapped[str] = mapped_column(String(32), default="pending")
