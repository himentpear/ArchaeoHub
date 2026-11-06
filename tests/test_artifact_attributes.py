"""Behavioral regressions for attribute binding, review and source changes."""
import copy
import json
from pathlib import Path

import pytest
from sqlalchemy import func, select

from src.artifact_attributes import (
    Candidate, SpecimenUnit, artifact_detail, collect_units,
    enrich_artifact_attributes, evidence_is_current, export_attributes,
    invalidate_candidates, project_artifact, rule_assertions, store_assertions,
    validate_model_assertions,
)
from src.db.models import AltoBlock, Artifact, Canvas, Manifest, ReviewTask, Revision, Work
from src.db.session import get_session, init_db
from src.model_router import ModelRouter
from src.review import resolve_review

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def library(tmp_path):
    engine = init_db("sqlite:///" + (tmp_path / "library.db").as_posix())
    with get_session(engine) as session:
        work = Work(slug="test", title="测试窑址")
        session.add(work)
        session.flush()
        manifest = Manifest(work_id=work.id, volume_number=1, label="第一卷")
        session.add(manifest)
        session.flush()
        canvas = Canvas(
            manifest_id=manifest.id, sequence_index=1, scan_index=1,
            printed_page_label="61", page_domain="body",
            width=1200, height=1600, iiif_image_service_id="test/image",
        )
        session.add(canvas)
        session.flush()
        for code in ["T1①∶1", "T1①∶2"]:
            session.add(Artifact(work_id=work.id, artifact_no=code))
        session.flush()
        ids = manifest.id, canvas.id, work.id
    yield engine, ids
    engine.dispose()


def add_block(session, canvas_id, text, order, kind="paragraph"):
    block = AltoBlock(
        canvas_id=canvas_id, alto_element_id=f"block-{order}", block_type=kind,
        page_zone="print_space", reading_order=order, column_index=0,
        x=100, y=100 + order * 50, w=900, h=40,
        content_text=text, raw_ocr=text,
    )
    session.add(block)
    session.flush()
    return block


def seed(session, ids):
    mid, cid, _ = ids
    add_block(session, cid, "一、青白瓷\n（1）碗\nAa型：敞口，圈足。", 1, "heading")
    block = add_block(session, cid,
        "标本1，T1①∶1，足径6、残高5厘米。外壁刻龙纹。\n"
        "标本2，T1①∶2，口径12、高16毫米。内壁有疑似莲\n纹。", 2)
    return block


def test_bind_each_specimen_and_shared_units_without_cross_contamination(library):
    engine, ids = library
    with get_session(engine) as session:
        seed(session, ids)
        summary = enrich_artifact_attributes(session, ids[0])
        arts = list(session.scalars(select(Artifact).order_by(Artifact.artifact_no)))
        first, second = arts
        assert first.artifact_type == second.artifact_type == "碗"
        assert first.material == "青白瓷" and first.typology == "Aa型"
        assert first.measurements == {
            "足径": {"unit": "cm", "value": 6.0}, "残高": {"unit": "cm", "value": 5.0},
        }
        assert second.measurements == {
            "口径": {"unit": "cm", "value": 1.2}, "高": {"unit": "cm", "value": 1.6},
        }
        assert first.decoration == ["外壁刻龙纹"]
        assert second.decoration == ["内壁有疑似莲\n纹"]
        assert first.review_status == "needs_review"
        assert summary["fields_changed"] == 12
        assert first.period is None
        for candidate in session.scalars(select(Candidate)):
            assert evidence_is_current(session, candidate)
            assert candidate.status == "pending"
        count = session.scalar(select(func.count()).select_from(Candidate))
        revisions = session.scalar(select(func.count()).select_from(Revision))
        again = enrich_artifact_attributes(session, ids[0])
        assert again["candidates_added"] == again["fields_changed"] == 0
        assert session.scalar(select(func.count()).select_from(Candidate)) == count
        assert session.scalar(select(func.count()).select_from(Revision)) == revisions


def test_scope_resets_and_no_dating_from_report_heading(library):
    engine, ids = library
    with get_session(engine) as session:
        cid = ids[1]
        add_block(session, cid, "一、青白瓷\n（1）碗\nAa型：圈足。", 1, "heading")
        add_block(session, cid, "标本，T1①∶1，高5厘米。", 2)
        add_block(session, cid, "第二章 宋元时期遗物\n2012～2017年考古发掘报告", 3, "heading")
        add_block(session, cid, "标本，T1①∶2，高6厘米。青白釉光亮。", 4)
        enrich_artifact_attributes(session, ids[0])
        arts = list(session.scalars(select(Artifact).order_by(Artifact.artifact_no)))
        assert arts[0].material == "青白瓷"
        assert arts[1].material is arts[1].period is arts[1].typology is arts[1].artifact_type is None


def test_measurement_ranges_remain_ranges_after_unit_conversion(library):
    engine, ids = library
    with get_session(engine) as session:
        add_block(session, ids[1], "标本，T1①∶1，口径0.05～0.06米，残高5毫米。", 1)
        enrich_artifact_attributes(session, ids[0])
        artifact = session.scalar(select(Artifact).where(Artifact.artifact_no == "T1①∶1"))
        assert artifact.measurements == {
            "口径": {"min": 5.0, "max": 6.0, "unit": "cm"},
            "残高": {"value": .5, "unit": "cm"},
        }


def test_caption_binds_local_types_not_ranges_or_other_groups(library):
    engine, ids = library
    with get_session(engine) as session:
        add_block(session, ids[1],
                  "图2.2.8 青白釉碗La～Le型\n"
                  "1～4. La型（T1①∶1） 5. Lb型（T1①∶2）", 1, "caption")
        enrich_artifact_attributes(session, ids[0])
        arts = list(session.scalars(select(Artifact).order_by(Artifact.artifact_no)))
        assert [a.artifact_type for a in arts] == ["碗", "碗"]
        assert [a.typology for a in arts] == ["La型", "Lb型"]
        assert all(a.material is None and a.measurements == {} for a in arts)


def test_ambiguous_multiple_codes_and_comparisons_are_not_attributed(library):
    engine, ids = library
    with get_session(engine) as session:
        add_block(session, ids[1], "标本，T1①∶1、T1①∶2，高5厘米。", 1)
        assert collect_units(session, ids[0]) == []
        add_block(session, ids[1], "标本，T1①∶1，口径5厘米。\n与T1①∶2相比，足径8厘米。", 2)
        enrich_artifact_attributes(session, ids[0])
        arts = list(session.scalars(select(Artifact).order_by(Artifact.artifact_no)))
        assert arts[0].measurements == {"口径": {"unit": "cm", "value": 5.0}}
        assert arts[1].measurements == {}


def test_misclassified_plate_caption_and_wrapped_figure_number(library):
    engine, ids = library
    with get_session(engine) as session:
        add_block(session, ids[1], "一、单色釉瓷器\n1. 青白釉瓷器\n（1）碗\nAa型：敞口。", 1, "heading")
        add_block(session, ids[1],
                  "标本，T1①∶1，口径5厘米（图\n2.2.1，1；图版2.2.1，1）。", 2)
        add_block(session, ids[1], "Ab型：圈足。\n标本，T1①∶2，口径6厘米。", 3)
        enrich_artifact_attributes(session, ids[0])
        arts = list(session.scalars(select(Artifact).order_by(Artifact.artifact_no)))
        assert [a.artifact_type for a in arts] == ["碗", "碗"]
        assert [a.material for a in arts] == ["青白釉瓷", "青白釉瓷"]
        assert [a.typology for a in arts] == ["Aa型", "Ab型"]
        assert arts[0].measurements["口径"]["value"] == 5
        # Older layout rules classified this plate label as a heading.
        add_block(session, ids[1], "1. 卵白釉罐Cb型Ⅳ式（T1①∶2）", 4, "heading")
        units = collect_units(session, ids[0])
        captions = [u for u in units if u.kind == "caption"]
        assert len(captions) == 1
        facts = {a["field_name"]: a["value"] for a in rule_assertions(captions[0])}
        assert facts == {"artifact_type": "罐", "typology": "Cb型Ⅳ式"}


def test_conflicting_values_do_not_overwrite_and_manual_columns_are_preserved(library):
    engine, ids = library
    with get_session(engine) as session:
        add_block(session, ids[1], "标本，T1①∶1，高5厘米。", 1)
        add_block(session, ids[1], "标本，T1①∶1，高6厘米。", 2)
        enrich_artifact_attributes(session, ids[0])
        artifact = session.scalar(select(Artifact).where(Artifact.artifact_no == "T1①∶1"))
        assert artifact.measurements == {}
        assert artifact.metadata_json["attribute_workflow"]["fields"]["measurements.高"]["status"] == "conflict"
        artifact.measurements = {"高": {"value": 7, "unit": "cm"}}
        project_artifact(session, artifact.id)
        assert artifact.measurements["高"]["value"] == 7
        assert artifact.metadata_json["attribute_workflow"]["fields"]["measurements.高"]["status"] == "external_value"


def test_review_reject_rerun_and_stale_revision(library):
    engine, ids = library
    with get_session(engine) as session:
        seed(session, ids)
        enrich_artifact_attributes(session, ids[0])
        candidate = session.scalar(select(Candidate).where(Candidate.field_name == "measurements.残高"))
        task = session.scalar(select(ReviewTask).where(ReviewTask.target_id == candidate.id))
        detail = artifact_detail(session, candidate.artifact_id)
        task_id, artifact_id = task.id, candidate.artifact_id
        old_revision = detail["artifact_revision"]
        resolve_review(
            session, task.id, status="rejected", actor="测试审校员", note="原页确认该候选不适用",
            expected_revision=1, expected_artifact_revision=old_revision,
        )
        assert "残高" not in session.get(Artifact, artifact_id).measurements
        enrich_artifact_attributes(session, ids[0])
        assert candidate.status == "rejected"
        assert "残高" not in session.get(Artifact, artifact_id).measurements
    with pytest.raises(ValueError, match="revision conflict"):
        with get_session(engine) as session:
            other = session.scalar(select(Candidate).where(Candidate.artifact_id == artifact_id, Candidate.status == "pending"))
            task = session.scalar(select(ReviewTask).where(ReviewTask.target_id == other.id))
            resolve_review(
                session, task.id, status="approved", actor="测试审校员", note="过期器物版本",
                expected_revision=2, expected_artifact_revision=old_revision,
            )


def test_approved_assertion_survives_new_conflict_and_source_edit_invalidates(library):
    engine, ids = library
    with get_session(engine) as session:
        seed(session, ids)
        enrich_artifact_attributes(session, ids[0])
        candidate = session.scalar(select(Candidate).where(Candidate.field_name == "measurements.残高"))
        task = session.scalar(select(ReviewTask).where(ReviewTask.target_id == candidate.id))
        artifact = session.get(Artifact, candidate.artifact_id)
        revision = artifact.metadata_json["attribute_workflow"]["revision"]
        resolve_review(session, task.id, status="approved", actor="测试审校员", note="对照原页确认残高",
                       expected_revision=1, expected_artifact_revision=revision)
        add_block(session, ids[1], "标本，T1①∶1，残高8厘米。", 3)
        enrich_artifact_attributes(session, ids[0])
        assert artifact.measurements["残高"]["value"] == 5
        assert artifact.metadata_json["attribute_workflow"]["fields"]["measurements.残高"]["status"] == "approved_with_conflict"
        # Correcting a shared heading invalidates even a later-page assertion.
        heading = session.scalar(select(AltoBlock).where(AltoBlock.reading_order == 1))
        heading.approved_text = "一、白瓷\n（1）盘\nBa型：平底。"
        assert invalidate_candidates(session, heading.id) > 0
        assert artifact.artifact_type is artifact.typology is artifact.material is None
        assert artifact.measurements["残高"]["value"] == 5
        source_block = session.scalar(select(AltoBlock).where(AltoBlock.reading_order == 2))
        source_block.approved_text = source_block.content_text.replace("残高5", "残高7")
        invalidate_candidates(session, source_block.id)
        assert candidate.status == "stale"
        assert artifact.measurements["残高"]["value"] == 8


def test_models_must_ground_target_attribute_labels_and_values(library):
    engine, ids = library
    with get_session(engine) as session:
        seed(session, ids)
        unit = collect_units(session, ids[0])[0]
        accepted = validate_model_assertions(unit, {"attributes": [
            {"field": "measurements.残高", "value": {"value": 5, "unit": "cm"}, "quote": "足径6、残高5厘米"},
            {"field": "artifact_type", "value": "碗", "quote": "碗"},
        ]}, provider="test", model="fake")
        assert len(accepted) == 2
        extra = validate_model_assertions(unit, {"attributes": [
            {"field": "typology", "value": "Aa型：敞口，圈足。", "quote": "Aa型：敞口，圈足。"},
        ]}, provider="test", model="fake")
        assert extra[0]["value"] == "Aa型"
        for item in [
            {"field": "measurements.高", "value": {"value": 5, "unit": "cm"}, "quote": "足径6、残高5厘米"},
            {"field": "measurements.足径", "value": {"value": 16, "unit": "cm"}, "quote": "足径6、残高5厘米"},
            {"field": "material", "value": "青铜", "quote": "外壁刻龙纹"},
            {"field": "period", "value": "宋代", "quote": "外壁刻龙纹"},
            {"field": "decoration", "value": "内壁有疑似莲纹", "quote": "内壁有疑似莲纹"},
        ]:
            with pytest.raises(ValueError):
                validate_model_assertions(unit, {"attributes": [item]}, provider="test", model="fake")


def test_separate_work_same_code_is_not_backfilled(library):
    engine, ids = library
    with get_session(engine) as session:
        seed(session, ids)
        other = Work(slug="other-site", title="另一窑址")
        session.add(other)
        session.flush()
        other_artifact = Artifact(work_id=other.id, artifact_no="T1①∶1")
        session.add(other_artifact)
        enrich_artifact_attributes(session, ids[0])
        assert other_artifact.measurements == {}
        assert not list(session.scalars(select(Candidate).where(Candidate.artifact_id == other_artifact.id)))


def test_candidate_only_then_projection_and_export(library, tmp_path):
    engine, ids = library
    with get_session(engine) as session:
        seed(session, ids)
        first = enrich_artifact_attributes(session, ids[0], materialize=False)
        artifact = session.scalar(select(Artifact).where(Artifact.artifact_no == "T1①∶1"))
        assert first["candidates_added"] > 0 and artifact.measurements == {}
        assert enrich_artifact_attributes(session, ids[0])["candidates_added"] == 0
        assert artifact.measurements
        export_attributes(session, ids[0], tmp_path)
        export = json.loads((tmp_path / "artifacts.json").read_text(encoding="utf-8"))
        assert export["artifacts"][0]["attribute_workflow"]["fields"]["measurements.残高"]["status"] == "pending"
        assert "source_sha256" in (tmp_path / "artifact_attribute_candidates.csv").read_text(encoding="utf-8-sig")


def test_attribute_model_cache_and_cloud_budget(monkeypatch, tmp_path):
    router = ModelRouter(tmp_path, allow_cloud=True, max_calls=1, max_total_tokens=10000)
    router.key = "fake-test-key"
    calls = []
    def response(messages, **kwargs):
        calls.append(messages)
        return {"choices": [{"message": {"content": json.dumps({"attributes": []})}, "finish_reason": "stop"}]}
    monkeypatch.setattr(router, "_cloud", response)
    first = router.extract_artifact_attributes("标本T1①∶1，高5厘米。", "T1①∶1", force_cloud=True)
    second = router.extract_artifact_attributes("标本T1①∶1，高5厘米。", "T1①∶1", force_cloud=True)
    assert first["result"] == {"attributes": []}
    assert second["cached"] and len(calls) == 1
    assert "fake-test-key" not in "".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("*.json"))


def test_http_attribute_review_has_evidence_versions_and_updated_export(library, tmp_path):
    import threading
    import httpx
    from src.server import make_server
    engine, ids = library
    with get_session(engine) as session:
        seed(session, ids)
        enrich_artifact_attributes(session, ids[0])
        candidate = session.scalar(select(Candidate).where(Candidate.field_name == "measurements.残高"))
        artifact_id, candidate_id = candidate.artifact_id, candidate.id
    server = make_server(tmp_path, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}", timeout=30) as client:
            detail = client.get("/api/artifact", params={"id": artifact_id}).json()
            via_candidate = client.get("/api/artifact", params={"candidate": candidate_id}).json()
            assert detail["id"] == via_candidate["id"]
            row = next(r for r in detail["candidates"] if r["id"] == candidate_id)
            assert row["source_current"] and row["evidence"]
            payload = {
                "task_id": row["task_id"], "actor": "API测试员", "note": "原页与尺寸类型核对",
                "status": "approved", "expected_revision": row["page_revision"],
                "expected_artifact_revision": detail["artifact_revision"],
            }
            response = client.post("/api/review", json=payload)
            assert response.status_code == 200, response.text
            updated = client.get("/api/artifact", params={"id": artifact_id}).json()
            assert updated["workflow"]["fields"]["measurements.残高"]["status"] == "approved"
            assert client.post("/api/review", json=payload).status_code == 409
            export = json.loads((tmp_path / "test" / "artifacts.json").read_text(encoding="utf-8"))
            assert any(a["attribute_workflow"]["fields"]["measurements.残高"]["status"] == "approved"
                       for a in export["artifacts"] if a["id"] == artifact_id)
            page = client.get("/").text
            assert "openArtifactAttributes" in page and "expected_artifact_revision" in page
    finally:
        server.shutdown()
        worker.join()
        server.server_close()
        server.engine.dispose()


def test_partial_model_validation_keeps_good_fields_and_records_rejections(library, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.artifact_attributes import run_model_attributes
    engine, ids = library
    with get_session(engine) as session:
        seed(session, ids)
        unit = collect_units(session, ids[0])[0]
    fake = SimpleNamespace(
        local_model=None, allow_cloud=True, key="fake", llm=None,
        extract_artifact_attributes=lambda *a, **kw: {
            "provider": "test", "model": "fake",
            "result": {"attributes": [
                {"field": "measurements.残高", "value": {"value": 5, "unit": "cm"}, "quote": "足径6、残高5厘米"},
                {"field": "period", "value": "明代", "quote": "年代为明代"},
            ]},
        },
    )
    monkeypatch.setattr(ModelRouter, "from_settings", lambda *a, **kw: fake)
    result = run_model_attributes(engine, ids[0], tmp_path, units=[unit], max_units=1)
    assert result["processed"] == 1 and result["partially_grounded_units"] == 1
    assert len(result["rejected_attributes"]) == 1 and not result["errors"]
    with get_session(engine) as session:
        artifact = session.get(Artifact, unit.artifact_id)
        assert artifact.measurements["残高"]["value"] == 5 and artifact.period is None
        assert session.scalar(select(ReviewTask).where(
            ReviewTask.risk_type == "artifact_attribute_model_failure"
        )).status == "pending"


def test_integer_and_float_measurements_from_different_providers_agree(library):
    engine, ids = library
    with get_session(engine) as session:
        seed(session, ids)
        enrich_artifact_attributes(session, ids[0])
        unit = collect_units(session, ids[0])[0]
        assertions = validate_model_assertions(unit, {"attributes": [
            {"field": "measurements.残高", "value": {"value": 5, "unit": "cm"}, "quote": "足径6、残高5厘米"},
        ]}, provider="test", model="fake")
        store_assertions(session, assertions, "http://example/iiif")
        session.flush()
        project_artifact(session, unit.artifact_id)
        artifact = session.get(Artifact, unit.artifact_id)
        assert artifact.measurements["残高"]["value"] == 5
        assert artifact.metadata_json["attribute_workflow"]["fields"]["measurements.残高"]["status"] == "pending"


def test_cli_dry_run_does_not_change_database_or_exports(library, tmp_path):
    import hashlib
    import subprocess
    import sys
    engine, ids = library
    with get_session(engine) as session:
        seed(session, ids)
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
    database = tmp_path / "library.db"
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/enrich_artifacts.py"), "--library", str(tmp_path),
         "--manifest", ids[0], "--dry-run"], capture_output=True, check=True,
    )
    summary = json.loads(result.stdout.decode("utf-8"))
    assert summary["dry_run"] and summary["assertions"] > 0
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    assert not (tmp_path / "backups").exists() and not (tmp_path / "test").exists()


def test_stale_projection_clears_in_candidate_only_run_and_can_regenerate(library):
    engine, ids = library
    with get_session(engine) as session:
        block = seed(session, ids)
        enrich_artifact_attributes(session, ids[0])
        artifact = session.scalar(select(Artifact).where(Artifact.artifact_no == "T1①∶1"))
        original = block.content_text
        block.approved_text = block.content_text.replace("残高5", "残高7")
        enrich_artifact_attributes(session, ids[0], materialize=False)
        assert "残高" not in artifact.measurements
        # The binding identifier shares the changed block, so its old type
        # assertion is conservatively invalidated as well.
        assert artifact.artifact_type is None
        block.approved_text = original
        enrich_artifact_attributes(session, ids[0])
        assert artifact.measurements["残高"]["value"] == 5


def test_real_three_pages_specimen_measurements(tmp_path):
    """Read real source text; never mutate the live library in a test."""
    import sqlite3
    source = ROOT / "data/processed/library/library.db"
    if not source.exists():
        pytest.skip("Local source corpus is not distributed with the repository")
    connection = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    rows = connection.execute("""
        select b.content_text from alto_blocks b join canvases c on c.id=b.canvas_id
        where c.manifest_id='38ef9921-7ae9-4b56-ab6a-1e887a68b1c3'
        and c.scan_index in (119,178,354) and b.block_type in ('paragraph','heading')
        order by c.scan_index,b.reading_order
    """).fetchall()
    connection.close()
    engine = init_db("sqlite:///" + (tmp_path / "sample.db").as_posix())
    try:
        with get_session(engine) as session:
            work = Work(slug="sample", title="真实三页样本")
            session.add(work);session.flush()
            manifest = Manifest(work_id=work.id, volume_number=1, label="第一卷")
            session.add(manifest);session.flush()
            canvas = Canvas(manifest_id=manifest.id, sequence_index=1, scan_index=1,
                            width=1200, height=1600, iiif_image_service_id="sample")
            session.add(canvas);session.flush()
            for code in ["T30H2∶301", "T3扩④a∶13", "T13H2∶602"]:
                session.add(Artifact(work_id=work.id, artifact_no=code))
            session.flush()
            for i, (text,) in enumerate(rows):
                add_block(session, canvas.id, text, i + 1)
            enrich_artifact_attributes(session, manifest.id)
            artifacts = {a.artifact_no: a for a in session.scalars(select(Artifact))}
            assert artifacts["T30H2∶301"].measurements["足径"]["value"] == 6
            assert artifacts["T30H2∶301"].measurements["残高"]["value"] == 5
            assert artifacts["T3扩④a∶13"].measurements["口径"]["value"] == 13.4
            assert artifacts["T3扩④a∶13"].measurements["高"]["value"] == 4.3
            assert artifacts["T13H2∶602"].measurements["足径"]["value"] == 5.2
            assert artifacts["T13H2∶602"].measurements["残高"]["value"] == 3.2
    finally:
        engine.dispose()
