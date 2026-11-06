"""Read-only provenance/consistency audit; does not estimate semantic accuracy."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.artifact_attributes import FIELDS, digest, normalize_code
from src.ingest.page_pipeline import atomic_json


def connect(path):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def text_hash(connection, table, columns):
    result = hashlib.sha256()
    for row in connection.execute(f"select {','.join(columns)} from {table} order by id"):
        result.update(json.dumps(list(row), ensure_ascii=False).encode())
    return result.hexdigest()


def audit(database, baseline=None):
    connection = connect(database)
    errors = []
    try:
        blocks = {}
        for row in connection.execute("""
            select b.*,c.manifest_id from alto_blocks b join canvases c on c.id=b.canvas_id
        """):
            text = next((row[k] for k in (
                "approved_text", "corrected_text", "normalized_ocr", "raw_ocr", "content_text"
            ) if row[k] is not None), "")
            blocks[row["id"]] = (text, digest(text), row["canvas_id"], row["manifest_id"])
        artifacts = {r["id"]: dict(r) for r in connection.execute("select * from artifacts")}
        manifests = {r["id"]: r["work_id"] for r in connection.execute("select id,work_id from manifests")}
        statuses, methods = Counter(), Counter()
        active, pieces = 0, 0
        for row in connection.execute("select * from artifact_attribute_candidates"):
            statuses[row["status"]] += 1
            methods[row["method"]] += 1
            if row["status"] not in {"pending", "approved"}:
                continue
            active += 1
            artifact = artifacts[row["artifact_id"]]
            if artifact["work_id"] != manifests[row["manifest_id"]]:
                errors.append({"id": row["id"], "kind": "cross_work_binding"})
            evidence = json.loads(row["evidence"])
            if not any(p["role"] == "binding" and normalize_code(p["quote"]) == normalize_code(artifact["artifact_no"]) for p in evidence):
                errors.append({"id": row["id"], "kind": "missing_specimen_binding"})
            for piece in evidence:
                pieces += 1
                source = blocks.get(piece["block_id"])
                if source is None or (
                    source[1] != piece["source_sha256"]
                    or source[0][piece["char_start"]:piece["char_end"]] != piece["quote"]
                    or source[2] != piece["canvas_id"] or source[3] != row["manifest_id"]
                    or piece["manifest_id"] != row["manifest_id"]
                ):
                    errors.append({"id": row["id"], "kind": "stale_or_mislocated_evidence"})
        coverage = Counter()
        field_status = Counter()
        projected_artifacts = 0
        for artifact in artifacts.values():
            artifact["decoration"] = json.loads(artifact["decoration"])
            artifact["measurements"] = json.loads(artifact["measurements"])
            metadata = json.loads(artifact["metadata"])
            populated = False
            for name in FIELDS:
                if artifact[name] not in (None, "", [], {}):
                    coverage[name] += 1
                    populated = True
            projected_artifacts += populated
            for name, state in metadata.get("attribute_workflow", {}).get("fields", {}).items():
                field_status[state["status"]] += 1
                actual = artifact["measurements"].get(name[13:]) if name.startswith("measurements.") else artifact[name]
                expected = [] if name == "decoration" and state["value"] is None else state["value"]
                if actual != expected:
                    errors.append({"id": artifact["id"], "field": name, "kind": "projection_mismatch"})
        preservation = {}
        if baseline:
            before = connect(baseline)
            try:
                for table, columns in [
                    ("alto_blocks", ["id", "raw_ocr", "normalized_ocr", "corrected_text", "approved_text", "content_text"]),
                    ("alto_lines", ["id", "raw_ocr", "normalized_ocr", "corrected_text", "approved_text", "content_text"]),
                ]:
                    old, current = text_hash(before, table, columns), text_hash(connection, table, columns)
                    preservation[table] = {"unchanged": old == current, "baseline_hash": old, "current_hash": current}
                    if old != current:
                        errors.append({"kind": "source_text_changed", "table": table})
            finally:
                before.close()
        integrity = connection.execute("pragma integrity_check").fetchone()[0]
        foreign_keys = [list(r) for r in connection.execute("pragma foreign_key_check")]
        if integrity != "ok" or foreign_keys:
            errors.append({"kind": "database_integrity"})
        return {
            "status": "passed" if not errors else "failed",
            "artifact_records": len(artifacts), "artifacts_with_projected_attributes": projected_artifacts,
            "candidate_status": dict(statuses), "candidate_methods": dict(methods),
            "active_candidates_checked": active, "evidence_pieces_checked": pieces,
            "field_coverage": {name: coverage[name] for name in FIELDS},
            "field_status": dict(field_status), "source_text_preservation": preservation,
            "integrity_check": integrity, "foreign_key_errors": foreign_keys, "errors": errors,
            "limitation": "Checks evidence/association/projection consistency, not human semantic accuracy or P/R/F1.",
        }
    finally:
        connection.close()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/processed/library/library.db"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/artifact_attribute_workflow_verification.json"))
    args = parser.parse_args()
    result = audit(args.database, args.baseline)
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
