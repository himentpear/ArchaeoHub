"""Specimen-bound table/body comparison with field-level source evidence."""
import re
from collections import defaultdict

from sqlalchemy import select

from src.alto.parser import AltoIngestor
from src.artifact_attributes import LABELS, collect_units, measurement_assertions, normalize_code, value_key
from src.artifact_identity import identity_state, work_namespace
from src.db.models import Artifact, Manifest, Work
from src.graphics.service import object_detail


def cell_measurements(cell):
    text = cell["text"]
    found = list(measurement_assertions(text))
    if found:
        return [(name, value) for name, value, _, _ in found]
    header = " ".join(value for value in cell.get("header_path", []) if isinstance(value, str))
    label = next((value for value in LABELS if value in header), None)
    unit = re.search(r"厘米|毫米|cm|mm|米|(?<![A-Za-z])m(?![A-Za-z])", header, re.I)
    if not label or not unit:
        return []
    return [(name, value) for name, value, _, _ in measurement_assertions(label + text + unit[0])]


def align_table(session, object_id, *, limit=200, offset=0):
    if not 1 <= limit <= 500:
        raise ValueError("Invalid row limit")
    if type(offset) is not int or offset<0:raise ValueError('Invalid row offset')
    detail = object_detail(session, object_id)
    if detail["category"] != "table" or detail["status"] in {"rejected", "superseded"}:
        raise ValueError("Select an active table")
    manifest = session.get(Manifest, detail["manifest_id"])
    namespace = work_namespace(manifest.work)
    by_code = defaultdict(list)
    for artifact, work in session.execute(select(Artifact, Work).join(Work, Artifact.work_id == Work.id)):
        if work_namespace(work) != namespace:
            continue
        identity = identity_state(artifact)
        for code in {normalize_code(artifact.artifact_no), normalize_code(identity["artifact_no"])}:
            by_code[code].append(artifact)
    grouped = defaultdict(list)
    for cell in detail["cells"]:
        if not cell["repeated_header"] and not cell.get("joined_into"):
            grouped[(cell["fragment_id"], cell["source_row"])].append(cell)
    rows = []
    for (fragment_id, row_index), cells in grouped.items():
        codes = sorted({normalize_code(code) for cell in cells for code in AltoIngestor.extract_artifact_codes(cell["text"])})
        if codes:
            rows.append({"fragment_id": fragment_id, "row": row_index, "codes": codes,
                         "cells": cells, "artifacts": by_code.get(codes[0], []) if len(codes) == 1 else []})
    selected = rows[offset:offset+limit]
    target_ids = {artifact.id for row in selected for artifact in row["artifacts"]}
    works = {artifact.work_id for row in selected for artifact in row["artifacts"]}
    descriptions = defaultdict(list)
    for report in session.scalars(select(Manifest).where(Manifest.work_id.in_(works))):
        for unit in collect_units(session, report.id):
            if unit.artifact_id in target_ids and unit.kind == "description":
                for name, value, start, end in measurement_assertions(unit.text):
                    descriptions[unit.artifact_id].append({"field": name, "value": value,
                        "evidence": unit.spans(start, end), "binding": unit.binding()})
    output = []
    for row in selected:
        identities = {(normalize_code(identity_state(a)["artifact_no"]), identity_state(a)["disambiguation"])
                      for a in row["artifacts"]}
        ambiguous = len(row["codes"]) > 1 or len(identities) > 1
        assertions = [{"cell_id": cell["id"], "field": name, "value": value, "anchors": cell["anchors"]}
                      for cell in row["cells"] for name, value in cell_measurements(cell)]
        comparisons = []
        if not ambiguous:
            for assertion in assertions:
                evidence = [item for artifact in row["artifacts"] for item in descriptions[artifact.id]
                            if item["field"] == assertion["field"]]
                for item in evidence:
                    comparisons.append({"field": assertion["field"], "table": assertion,
                        "body": item, "status": "agree" if value_key(assertion["value"]) == value_key(item["value"]) else "conflict"})
        output.append({"row": row["row"], "fragment_id": row["fragment_id"], "codes": row["codes"],
            "status": "ambiguous" if ambiguous else "linked" if row["artifacts"] else "unmatched",
            "artifact_ids": [a.id for a in row["artifacts"]],
            "cells": [{key:cell[key] for key in ("id", "text", "column", "header_path", "anchors")} for cell in row["cells"]],
            "table_measurements": assertions, "comparisons": comparisons,
            "conflicts": sum(item["status"] == "conflict" for item in comparisons)})
    return {"object_id": object_id, "published": detail["published"], "table_revision": detail["revision"],
            "total_rows": len(grouped), "identifier_rows": len(rows), "processed_rows": len(output),
            "remaining_rows": max(0, len(rows) - offset - limit), "offset":offset,"rows": output,
            "conflicts": sum(row["conflicts"] for row in output),
            "ambiguous_rows": sum(row["status"] == "ambiguous" for row in output)}
