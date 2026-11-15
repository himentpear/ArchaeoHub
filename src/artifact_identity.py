"""Audited editorial numbers and identity scopes over source-bound artifacts."""
from __future__ import annotations

import copy

from sqlalchemy import select, update

from src.db.models import Artifact, ArtifactOccurrence, Canvas, Revision


def work_namespace(work):
    return (work.metadata_json or {}).get("source_namespace") or work.site_name or "legacy-corpus"


def identity_state(artifact):
    identity = (artifact.metadata_json or {}).get("identity", {})
    return {
        "artifact_no": identity.get("corrected_no") or artifact.artifact_no,
        "original_artifact_no": artifact.artifact_no,
        "disambiguation": identity.get("disambiguation", ""),
        "revision": identity.get("revision", 0),
    }


def identity_detail(session, artifact_id):
    artifact = session.get(Artifact, artifact_id)
    if artifact is None:
        raise LookupError("Artifact not found")
    history = list(session.scalars(select(Revision).where(
        Revision.entity_type == "artifact_identity", Revision.entity_id == artifact_id,
    ).order_by(Revision.revision_no.desc())))
    occurrences = session.execute(select(ArtifactOccurrence, Canvas).join(
        Canvas, ArtifactOccurrence.canvas_id == Canvas.id,
    ).where(ArtifactOccurrence.artifact_id == artifact_id).order_by(Canvas.scan_index)).all()
    return {
        "id": artifact_id, "work_id": artifact.work_id, **identity_state(artifact),
        "occurrences": [{
            "id": occurrence.id, "manifest_id": canvas.manifest_id,
            "canvas_id": canvas.id, "scan_index": canvas.scan_index,
            "printed_page": canvas.printed_page_label, "xywh": occurrence.target_xywh,
            "snippet": occurrence.context_snippet or "",
        } for occurrence, canvas in occurrences],
        "history": [{
            "id": row.id, "revision": row.revision_no, "actor": row.actor,
            "reason": row.reason, "before": row.before_state, "after": row.after_state,
            "created_at": row.created_at.isoformat(),
        } for row in history],
    }


def edit_identity(session, artifact_id, *, expected_revision, actor, note,
                  artifact_no=None, disambiguation="", undo_revision_id=None):
    if not isinstance(actor, str) or not 1 <= len(actor.strip()) <= 128:
        raise ValueError("填写审校人，长度上限为128字")
    if not isinstance(note, str) or not 1 <= len(note.strip()) <= 255:
        raise ValueError("填写修改依据，长度上限为255字")
    artifact = session.get(Artifact, artifact_id)
    if artifact is None:
        raise LookupError("Artifact not found")
    metadata = copy.deepcopy(artifact.metadata_json or {})
    before = identity_state(artifact)
    if type(expected_revision) is not int or before["revision"] != expected_revision:
        raise ValueError("编号版本已更新，请重新打开后修改")
    if undo_revision_id:
        previous = session.get(Revision, undo_revision_id)
        if (previous is None or previous.entity_type != "artifact_identity"
                or previous.entity_id != artifact_id or previous.revision_no != expected_revision):
            raise ValueError("只能撤销当前编号版本")
        artifact_no = previous.before_state["artifact_no"]
        disambiguation = previous.before_state.get("disambiguation", "")
    if not isinstance(artifact_no, str) or not 1 <= len(artifact_no.strip()) <= 128:
        raise ValueError("编号应为1至128字")
    if not isinstance(disambiguation, str) or len(disambiguation.strip()) > 128:
        raise ValueError("消歧标记长度上限为128字")
    after = {
        "artifact_no": artifact_no.strip(), "original_artifact_no": artifact.artifact_no,
        "disambiguation": disambiguation.strip(), "revision": expected_revision + 1,
    }
    new_metadata = {**metadata, "identity": {
        "corrected_no": after["artifact_no"], "disambiguation": after["disambiguation"],
        "revision": after["revision"],
    }}
    claimed = session.execute(update(Artifact).where(
        Artifact.id == artifact_id, Artifact.metadata_json == metadata,
    ).values(metadata_json=new_metadata).execution_options(synchronize_session=False))
    if claimed.rowcount != 1:
        raise ValueError("器物记录已更新，请重新打开后修改")
    artifact.metadata_json = new_metadata
    session.add(Revision(
        entity_type="artifact_identity", entity_id=artifact_id,
        revision_no=after["revision"], actor=actor.strip(), reason=note.strip(),
        before_state=before, after_state=after,
    ))
    session.flush()
    return {"id": artifact_id, **after}


def find_identity_issues(index, query="", manifest_id="all", *, page=1, limit=50):
    """Flag conflicting assertions for review, retaining every evidence record."""
    if page < 1 or not 1 <= limit <= 100:
        raise ValueError("Invalid issue pagination")
    issues = []
    for entry in index["cross_artifacts"]:
        if manifest_id != "all" and manifest_id not in entry["manifest_ids"]:
            continue
        if query and query.casefold() not in entry["_norm"]:
            continue
        records = entry["attribute_records"]
        conflicts = {}
        for field in ("artifact_type", "excavation_unit", "stratum"):
            values = sorted({r[field] for r in records if r.get(field)})
            if len(values) > 1:
                conflicts[field] = values
        for record in records:
            for field, status in record["field_status"].items():
                if status in {"conflict", "approved_with_conflict"}:
                    conflicts.setdefault(field, []).append("属性候选存在冲突")
        if conflicts:
            issues.append({
                "code": entry["code"], "source_namespace": entry["source_namespace"],
                "disambiguation": entry.get("disambiguation", ""),
                "conflicts": conflicts, "records": records,
                "occurrences": entry["occurrences"],
            })
    offset = (page - 1) * limit
    return {"total": len(issues), "page": page, "limit": limit, "items": issues[offset:offset + limit]}
