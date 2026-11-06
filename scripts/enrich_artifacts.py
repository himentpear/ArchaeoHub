"""Incremental artifact attribute backfill; never repeats PDF/OCR ingestion."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine, select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.artifact_attributes import (
    collect_units, enrich_artifact_attributes, export_attributes,
    rule_assertions, run_model_attributes,
)
from src.db.models import Manifest
from src.db.session import get_session, init_db
from src.ingest.page_pipeline import atomic_json, job_lock


def backup(database: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    target = sqlite3.connect(destination)
    try:
        source.backup(target)
        if target.execute("pragma integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Attribute backfill backup failed integrity check")
    finally:
        target.close()
        source.close()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("data/processed/library"))
    parser.add_argument("--manifest")
    parser.add_argument("--dry-run", action="store_true", help="Read-only preview; creates no tables or files")
    parser.add_argument("--candidates-only", action="store_true", help="Store candidates without populating Artifact columns")
    parser.add_argument("--models", action="store_true", help="Also run bounded model extraction")
    parser.add_argument("--provider", choices=["auto", "cloud"], default="auto")
    parser.add_argument("--max-model-units", type=int, default=5)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--scan-index", type=int, action="append", help="Restrict to selected PDF page numbers")
    args = parser.parse_args()
    if args.max_model_units < 0 or any(i < 1 for i in args.scan_index or []):
        parser.error("Limits and page numbers must be nonnegative/positive")
    library = args.library.resolve()
    database = library / "library.db"
    if not database.is_file():
        parser.error("library.db does not exist")
    if args.dry_run:
        # Bypass init_db and its schema/WAL setup entirely.
        engine = create_engine("sqlite:///file:" + database.as_posix() + "?mode=ro&uri=true")
        try:
            with get_session(engine) as session:
                query = select(Manifest).order_by(Manifest.id)
                if args.manifest:
                    query = query.where(Manifest.id == args.manifest)
                documents = list(session.scalars(query))
                if not documents:
                    raise SystemExit("No matching manifest")
                for manifest in documents:
                    units = collect_units(session, manifest.id)
                    if args.scan_index:
                        units = [u for u in units if u.parts[0]["scan_index"] in args.scan_index]
                    assertions = [a for u in units for a in rule_assertions(u)]
                    print(json.dumps({
                        "manifest_id": manifest.id, "dry_run": True,
                        "units": len(units), "assertions": len(assertions),
                        "by_field": {name: sum(a["field_name"].split(".")[0] == name for a in assertions)
                                     for name in sorted({a["field_name"].split(".")[0] for a in assertions})},
                        "examples": assertions[:3],
                    }, ensure_ascii=False), flush=True)
        finally:
            engine.dispose()
        return
    with job_lock(library / "artifact_attributes.lock"):
        snapshot = library / "backups" / f"before-attributes-{time.time_ns()}.db"
        backup(database, snapshot)  # Before any additive schema initialization.
        engine = init_db("sqlite:///" + database.as_posix())
        try:
            with get_session(engine) as session:
                query = select(Manifest).order_by(Manifest.id)
                if args.manifest:
                    query = query.where(Manifest.id == args.manifest)
                documents = [(m.id, m.work.slug) for m in session.scalars(query)]
            if not documents:
                raise SystemExit("No matching manifest")
            summaries = []
            for manifest_id, slug in documents:
                run_dir = library / slug
                run_dir.mkdir(parents=True, exist_ok=True)
                with get_session(engine) as session:
                    units = collect_units(session, manifest_id) if args.scan_index else None
                    if units is not None:
                        units = [u for u in units if u.parts[0]["scan_index"] in args.scan_index]
                    result = enrich_artifact_attributes(
                        session, manifest_id, units=units, materialize=not args.candidates_only,
                    )
                if args.models:
                    result["model_stage"] = run_model_attributes(
                        engine, manifest_id, run_dir, settings_path=args.settings,
                        max_units=args.max_model_units, force_cloud=args.provider == "cloud",
                        materialize=not args.candidates_only, units=units,
                    )
                with get_session(engine) as session:
                    export_attributes(session, manifest_id, run_dir)
                result.update(manifest_id=manifest_id, pre_backfill_backup=str(snapshot))
                atomic_json(run_dir / "artifact_attribute_stage.json", result)
                summaries.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            atomic_json(library / "artifact_attribute_summary.json", {
                "documents": summaries, "backup": str(snapshot),
                "notice": "自动候选不等于人工标准答案，字段审校状态保存在attribute_workflow及候选表。",
            })
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
