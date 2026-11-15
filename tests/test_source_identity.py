"""Regressions for evidence labels and editable specimen identities."""
import json
import threading
import io

import httpx
import pytest
from sqlalchemy import select

from src.db.models import AltoBlock, AltoLine, Artifact, ArtifactOccurrence, Canvas, Manifest, Work
from src.db.session import get_session, init_db
from src.server import make_server
from src.source_types import source_type
from PIL import Image


def test_numbered_captions_are_recognized_in_legacy_paragraphs():
    assert source_type("paragraph", text="1. Ca型Ⅰ式（T29H3③∶36） 2. Ca型Ⅱ式（T10⑧∶300）") == "caption"
    assert source_type("heading", text="4. 匣钵Aa型Ⅲ式（T10⑧∶300）") == "caption"
    assert source_type("paragraph", text="标本T10⑧∶300，可复原。高2厘米。") == "body"


@pytest.fixture
def library(tmp_path):
    engine = init_db("sqlite:///" + (tmp_path / "library.db").as_posix())
    identifiers = []
    with get_session(engine) as session:
        for index in range(2):
            work = Work(slug=f"source-test-{index}", title=f"测试第{index + 1}卷",
                        metadata_json={"source_namespace": "identity-test"})
            session.add(work)
            session.flush()
            manifest = Manifest(work_id=work.id, volume_number=index + 1, label=work.title, total_pages=1)
            session.add(manifest)
            session.flush()
            canvas = Canvas(manifest_id=manifest.id, sequence_index=1, scan_index=1,
                            page_domain="body", width=1200, height=1600, iiif_image_service_id="test/image")
            session.add(canvas)
            session.flush()
            artifact = Artifact(work_id=work.id, artifact_no="T1①∶1", artifact_type=["碗", "盘"][index])
            session.add(artifact)
            session.flush()
            for order, kind in enumerate(["paragraph", "caption", "table", "graphic_text"], 1):
                content = f"T1①∶1 {kind} 来源测试"
                block = AltoBlock(canvas_id=canvas.id, alto_element_id=f"b{order}", block_type=kind,
                                  page_zone="print_space", reading_order=order, column_index=0,
                                  x=100, y=order * 100, w=600, h=40, raw_ocr=content, content_text=content)
                session.add(block)
                session.flush()
                line = AltoLine(block_id=block.id, canvas_id=canvas.id, alto_element_id=f"l{order}",
                                line_index=order, x=100, y=order * 100, w=600, h=40,
                                raw_ocr=content, content_text=content)
                session.add(line)
                session.add(ArtifactOccurrence(artifact_id=artifact.id, canvas_id=canvas.id,
                    layout_block_id=block.id, role="mentioned", context_snippet=content, target_xywh=block.xywh))
            identifiers.append({"artifact": artifact.id, "manifest": manifest.id})
    engine.dispose()
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}", timeout=20) as client:
            yield client, identifiers, tmp_path, server.engine
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        server.engine.dispose()


def test_labels_follow_blocks_on_same_body_page(library):
    client, identifiers, _, _ = library
    data = client.get("/api/search", params={"q": "来源测试", "manifest": identifiers[0]["manifest"]}).json()
    assert [hit["source_type"] for hit in data["items"]] == ["body", "caption", "table", "unknown"]
    artifacts = client.get("/api/cross_index", params={"cross_only": "0"}).json()["artifacts"]
    assert len(artifacts) == 1
    assert {o["source_type"] for o in artifacts[0]["occurrences"]} == {"body", "caption", "table", "unknown"}


def test_identity_edits_are_versioned_searchable_and_exported(library):
    client, identifiers, root, engine = library
    identifier = identifiers[0]["artifact"]
    client.get("/api/cross_index")  # Prime the cache before editing.
    response = client.post("/api/artifact_identity", json={"artifact_id": identifier,
        "expected_revision": 0, "actor": "测试员", "note": "对照原页纠正误识编号",
        "artifact_no": "T1①∶2", "disambiguation": "出土记录甲"})
    assert response.status_code == 200, response.text
    assert response.json()["export_errors"] == []
    corrected = client.get("/api/cross_index", params={"q": "T1①:2", "cross_only": "0"}).json()
    assert corrected["total_artifacts"] == 1
    assert corrected["artifacts"][0]["disambiguation"] == "出土记录甲"
    original = client.get("/api/cross_index", params={"q": "T1①:1", "cross_only": "0"}).json()
    assert original["total_artifacts"] == 2
    with get_session(engine) as session:
        assert session.get(Artifact, identifier).artifact_no == "T1①∶1"
        assert len(list(session.scalars(select(ArtifactOccurrence).where(ArtifactOccurrence.artifact_id == identifier)))) == 4
    exported = json.loads((root / "source-test-0" / "artifacts.json").read_text(encoding="utf-8"))
    assert exported["artifacts"][0]["artifact_no"] == "T1①∶2"
    assert exported["artifacts"][0]["original_artifact_no"] == "T1①∶1"
    detail = client.get("/api/artifact_identity", params={"id": identifier}).json()
    assert detail["revision"] == 1 and len(detail["history"]) == 1
    stale = client.post("/api/artifact_identity", json={"artifact_id": identifier,
        "expected_revision": 0, "actor": "其他测试员", "note": "旧版本提交", "artifact_no": "T1①∶3"})
    assert stale.status_code == 409
    undo = client.post("/api/artifact_identity", json={"artifact_id": identifier,
        "expected_revision": 1, "actor": "测试员", "note": "撤销测试修改",
        "undo_revision_id": detail["history"][0]["id"]})
    assert undo.status_code == 200, undo.text
    assert undo.json()["artifact_no"] == "T1①∶1" and undo.json()["revision"] == 2


def test_conflicts_and_disambiguation_do_not_merge_distinct_identities(library):
    client, identifiers, _, _ = library
    issues = client.get("/api/artifact_issues").json()
    assert issues["total"] == 1 and issues["items"][0]["conflicts"]["artifact_type"] == ["盘", "碗"]
    response = client.post("/api/artifact_identity", json={"artifact_id": identifiers[0]["artifact"],
        "expected_revision": 0, "actor": "测试员", "note": "核实同号异物", "artifact_no": "T1①∶1",
        "disambiguation": "碗"})
    assert response.status_code == 200
    assert client.get("/api/cross_index", params={"cross_only": "0"}).json()["total_artifacts"] == 2
    selected = client.get("/api/cross_index", params={"cross_only": "0", "artifact": identifiers[0]["artifact"]}).json()
    assert selected["total_artifacts"] == 1 and selected["artifacts"][0]["disambiguation"] == "碗"
    assert client.get("/api/artifact_issues").json()["total"] == 0


def test_identity_validation_and_origin_check(library):
    client, identifiers, _, _ = library
    data = {"artifact_id": identifiers[0]["artifact"], "expected_revision": 0,
            "actor": "测试员", "note": "编号测试", "artifact_no": "T1①∶2"}
    assert client.post("/api/artifact_identity", json=data, headers={"Origin": "https://unrelated.example"}).status_code == 403
    data["note"] = ""
    assert client.post("/api/artifact_identity", json=data).status_code == 409
    assert client.get("/api/artifact_identity", params={"id": "missing"}).status_code == 404


def test_library_audit_reports_missing_originals_for_every_document(library):
    from scripts.verify_library import verify
    _, _, root, _ = library
    result = verify(root)
    assert not result["passed"] and len(result["documents"]) == 2
    for document in result["documents"]:
        assert not document["source_preserved"]
        assert {"error": "missing_source_pdf"} in document["errors"]


def test_precise_crop_saves_coordinates_image_and_revision(library):
    client, identifiers, root, engine = library
    with get_session(engine) as session:
        occurrence = session.scalar(select(ArtifactOccurrence).where(ArtifactOccurrence.artifact_id == identifiers[0]["artifact"]))
        canvas = session.get(Canvas, occurrence.canvas_id)
        path = root / "source-image.jpg"
        Image.new("RGB", (120, 160), "white").save(path)
        canvas.image_storage_path = str(path)
        canvas_id = canvas.id
    source = client.get("/api/artifact_regions", params={"canvas": canvas_id}).json()
    payload = {"artifact_id": identifiers[0]["artifact"], "canvas_id": canvas_id,
        "xywh": [100, 200, 120, 80], "view_label": "口沿正视", "media_type": "line_drawing",
        "actor": "测试员", "note": "对照原图选择口沿区域", "expected_image_hash": source["canvas"]["image_hash"]}
    response = client.post("/api/artifact_regions", json=payload)
    assert response.status_code == 200, response.text
    region = response.json()
    image = client.get(region["image_url"])
    assert image.status_code == 200
    assert Image.open(io.BytesIO(image.content)).size == (120, 80)
    indexed = client.get("/api/cross_index", params={"cross_only": "0"}).json()["artifacts"][0]
    assert any(item.get("is_precise_crop") for item in indexed["occurrences"])
    stale = client.post("/api/artifact_regions", json={**payload, "region_id": region["id"], "expected_revision": 0})
    assert stale.status_code == 409
    bad = client.post("/api/artifact_regions", json={**payload, "xywh": [-1, 0, 100, 100]})
    assert bad.status_code == 409
    changed_source = client.post("/api/artifact_regions", json={**payload, "expected_image_hash": "old-image"})
    assert changed_source.status_code == 409
    withdrawn = client.post("/api/artifact_regions", json={**payload, "region_id": region["id"],
        "expected_revision": 1, "status": "rejected"})
    assert withdrawn.status_code == 200
    assert client.get(region["image_url"]).status_code == 404


def test_table_body_comparison_preserves_units_and_ambiguous_rows(library):
    from src.graphics.models import GraphicRun, GraphicObject, GraphicFragment, GraphicCell
    client, identifiers, _, engine = library
    with get_session(engine) as session:
        occurrence = session.scalar(select(ArtifactOccurrence).where(ArtifactOccurrence.artifact_id == identifiers[0]["artifact"]))
        canvas = session.get(Canvas, occurrence.canvas_id)
        text = "标本T1①∶1，高5厘米。"
        block = AltoBlock(canvas_id=canvas.id, alto_element_id="measure-body", block_type="paragraph",
            page_zone="print_space", reading_order=10, column_index=0, x=100, y=600, w=600, h=60,
            raw_ocr=text, content_text=text)
        session.add(block)
        run = GraphicRun(manifest_id=canvas.manifest_id, fingerprint="test", status="complete", published=False,
                         config={"pages": [1]})
        session.add(run)
        session.flush()
        table = GraphicObject(run_id=run.id, manifest_id=canvas.manifest_id, category="table", label="测量表")
        session.add(table)
        session.flush()
        fragment = GraphicFragment(object_id=table.id, canvas_id=canvas.id, source_key="test-table",
            bbox=[0,0,800,400], source_hash="a" * 64, crop_path="unused.jpg")
        session.add(fragment)
        session.flush()
        session.add(GraphicCell(fragment_id=fragment.id, row_index=0, column_index=0, raw_text="T1①∶1"))
        measurement = GraphicCell(fragment_id=fragment.id, row_index=0, column_index=1,
                                  raw_text="60", header_path=["高（毫米）"])
        session.add(measurement)
        session.add(GraphicCell(fragment_id=fragment.id, row_index=1, column_index=0, raw_text="T1①∶1、T1①∶2"))
        session.flush()
        object_id, cell_id = table.id, measurement.id
    result = client.get("/api/table_alignment", params={"id": object_id})
    assert result.status_code == 200, result.text
    comparison = result.json()
    assert comparison["identifier_rows"] == 2 and comparison["ambiguous_rows"] == 1
    assert comparison["conflicts"] == 1
    field = comparison["rows"][0]["comparisons"][0]
    assert field["table"]["value"] == {"unit": "cm", "value": 6}
    assert field["body"]["value"] == {"unit": "cm", "value": 5}
    assert field["body"]["evidence"][0]["quote"] == "高5厘米"
    with get_session(engine) as session:
        session.get(GraphicCell, cell_id).corrected_text = "50"
    updated = client.get("/api/table_alignment", params={"id": object_id}).json()
    assert updated["conflicts"] == 0 and updated["rows"][0]["comparisons"][0]["status"] == "agree"
