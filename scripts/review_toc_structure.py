"""Review existing/newly scanned report hierarchy with a strictly local model."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import select
from scripts.enrich_artifacts import backup
from src.db.models import Manifest
from src.db.session import get_session, init_db
from src.iiif.generator import IiifBuilder
from src.ingest.page_pipeline import atomic_json
from src.toc_structure import export_toc_structure, run_local_toc_review, toc_view


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("data/processed/library"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--start-scan", type=int)
    parser.add_argument("--end-scan", type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.start_scan is not None and args.start_scan < 1:
        parser.error("--start-scan must be positive")
    if args.end_scan is not None and args.end_scan < (args.start_scan or 1):
        parser.error("--end-scan must follow --start-scan")
    root = args.library.resolve()
    database = root / "library.db"
    if not database.is_file():
        parser.error("library.db not found")
    snapshot = root / "backups" / f"before-local-toc-{time.time_ns()}.db"
    backup(database, snapshot)
    engine = init_db("sqlite:///" + database.as_posix())
    try:
        with get_session(engine) as session:
            manifest = session.get(Manifest, args.manifest)
            if manifest is None:
                parser.error("Unknown manifest")
            run_dir = root / manifest.work.slug
        result = run_local_toc_review(
            engine, args.manifest, run_dir, settings_path=args.settings, model_path=args.model,
            max_nodes=args.max_nodes, batch_size=args.batch_size, max_batches=args.max_batches,
            start_scan=args.start_scan, end_scan=args.end_scan,
        )
        result["backup"] = str(snapshot)
        with get_session(engine) as session:
            export_toc_structure(session, args.manifest, run_dir)
            atomic_json(run_dir / "manifest.json", IiifBuilder(session, "http://127.0.0.1:8001/iiif").build_manifest(args.manifest))
            atomic_json(run_dir / "toc_model_manifest.json", IiifBuilder(session, "http://127.0.0.1:8001/iiif").build_manifest(args.manifest, toc_source="local_model"))
            view = toc_view(session, args.manifest, source="local_model")
        if args.report:
            atomic_json(args.report, {"summary": result, "review": view["run"]})
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
