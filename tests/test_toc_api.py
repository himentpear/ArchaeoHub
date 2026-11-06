"""HTTP regressions for proposed local-model chapter trees and baseline views."""
from __future__ import annotations

import copy
import sys
import threading
import types

import httpx
import pytest

from src.db.models import Canvas, Manifest, Work
from src.db.session import get_session, init_db
from src.server import _structure_toc_items, make_server


@pytest.fixture
def toc_service(tmp_path, monkeypatch):
    engine = init_db("sqlite:///" + (tmp_path / "library.db").as_posix())
    with get_session(engine) as session:
        work = Work(slug="toc-http-test", title="目录测试卷")
        session.add(work)
        session.flush()
        manifest = Manifest(work_id=work.id, volume_number=1, label="第一卷", total_pages=4)
        session.add(manifest)
        session.flush()
        pages = []
        for number in range(1, 5):
            canvas = Canvas(
                manifest_id=manifest.id, sequence_index=number, scan_index=number,
                printed_page_label=str(number), width=1200, height=1600,
                iiif_image_service_id=f"test/image/{number}",
            )
            session.add(canvas)
            session.flush()
            pages.append(canvas.id)
        manifest_id = manifest.id
    engine.dispose()

    def node(identifier, parent, level, title, number, method):
        return {
            "id": identifier, "parent_id": parent, "level": level,
            "title": title, "canvas_id": pages[number - 1], "scan_index": number,
            "printed_page": str(number), "xywh": "100,100,600,50",
            "source_method": method,
            "review_status": "pending" if method == "local_model" else "baseline",
            "section_number": None,
        }

    # A title containing a colon is rejected by the old hard filter. A local
    # model's source-grounded keep decision must not be silently hidden by it.
    model_items = [
        node("chapter-a", None, 1, "第一章：绪论", 1, "local_model"),
        node("child-a", "chapter-a", 2, "一、研究材料", 2, "local_model"),
        node("chapter-b", None, 1, "第二章 相关分析", 3, "auto_heading"),
        node("child-b", "chapter-b", 2, "一、分析方法", 4, "auto_heading"),
    ]
    rule_items = copy.deepcopy(model_items)
    rule_items[0]["title"] = "第一章 绪论"
    for row in rule_items:
        row["source_method"], row["review_status"] = "auto_heading", "baseline"
    state = {
        "run": {
            "id": "local-run-1", "status": "completed_with_review",
            "model": "local-test-model.gguf", "processed": 2, "total": 4,
            "changes": 2, "remaining": 2,
        },
        "calls": [],
    }

    def fake_toc_view(session, requested_id, source="auto"):
        state["calls"].append((requested_id, source))
        if source not in {"auto", "rules", "local_model"}:
            raise ValueError("Unknown TOC source")
        if session.get(Manifest, requested_id) is None:
            raise LookupError("Manifest not found")
        return {
            "items": copy.deepcopy(rule_items if source == "rules" else model_items),
            "run": None if source == "rules" else copy.deepcopy(state["run"]),
        }

    # The core model runner is developed independently. Mock only its view
    # contract so these HTTP tests never load weights or invoke inference.
    module = types.ModuleType("src.toc_structure")
    module.toc_view = fake_toc_view
    monkeypatch.setitem(sys.modules, "src.toc_structure", module)
    server = make_server(tmp_path, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}", timeout=20) as client:
            yield client, manifest_id, state
    finally:
        server.shutdown()
        worker.join()
        server.server_close()
        server.engine.dispose()


def test_toc_structure_endpoint_is_live_and_preserves_run_coverage(toc_service):
    client, manifest_id, state = toc_service
    first = client.get("/api/toc_structure", params={"manifest": manifest_id, "source": "local_model"})
    assert first.status_code == 200, first.text
    data = first.json()
    assert data["run"] == state["run"]
    assert [item["id"] for item in data["items"]] == ["chapter-a", "child-a", "chapter-b", "child-b"]
    assert data["items"][1]["parent_id"] == "chapter-a"
    assert data["items"][0]["review_status"] == "pending"
    assert state["calls"][-1] == (manifest_id, "local_model")
    state["run"]["processed"], state["run"]["remaining"] = 3, 1
    second = client.get("/api/toc_structure", params={"manifest": manifest_id})
    assert second.status_code == 200, second.text
    assert second.json()["run"]["processed"] == 3
    assert second.json()["run"]["remaining"] == 1
    assert state["calls"][-1] == (manifest_id, "auto")


@pytest.mark.parametrize("endpoint", ["/api/toc_structure", "/api/toc"])
def test_toc_endpoints_reject_unknown_source(toc_service, endpoint):
    client, manifest_id, _ = toc_service
    response = client.get(endpoint, params={"manifest": manifest_id, "source": "untrusted-mode"})
    assert response.status_code == 400


def test_toc_structure_unknown_manifest_returns_not_found(toc_service):
    client, _, _ = toc_service
    assert client.get("/api/toc_structure", params={"manifest": "missing-manifest"}).status_code == 404


def test_legacy_toc_array_keeps_model_decisions_parent_links_and_order(toc_service):
    client, manifest_id, _ = toc_service
    response = client.get("/api/toc", params={"manifest": manifest_id, "max_level": 0})
    assert response.status_code == 200, response.text
    data = response.json()
    assert isinstance(data, list)
    assert [item["id"] for item in data] == ["chapter-a", "child-a", "chapter-b", "child-b"]
    assert data[0]["title"] == "第一章：绪论"
    assert data[0]["source_method"] == "local_model"
    assert data[0]["review_status"] == "pending"
    assert data[1]["parent_id"] == data[0]["id"]
    assert data[3]["parent_id"] == data[2]["id"]
    assert all(item["manifest_id"] == manifest_id for item in data)


def test_rule_switch_returns_compatible_array_without_model_approval(toc_service):
    client, manifest_id, state = toc_service
    response = client.get("/api/toc", params={"manifest": manifest_id, "source": "rules", "max_level": 0})
    assert response.status_code == 200, response.text
    data = response.json()
    assert [item["title"] for item in data] == [
        "第一章 绪论", "一、研究材料", "第二章 相关分析", "一、分析方法",
    ]
    assert all(item["source_method"] == "auto_heading" for item in data)
    assert all(item["review_status"] == "baseline" for item in data)
    assert (manifest_id, "rules") in state["calls"]
    fresh = client.get("/api/toc_structure", params={"manifest": manifest_id, "source": "rules"}).json()
    assert fresh["run"] is None


def test_existing_manual_title_is_not_hidden_by_rule_filter():
    doc = {"id":"manual-doc","title":"人工目录卷","short_title":"人工目录卷"}
    raw = {
        "id":"manual-chapter","parent_id":None,"title":"第一章：人工确认标题",
        "level":1,"source_method":"manual","review_status":"baseline",
        "canvas_id":"page-1","scan_index":1,"printed_page":"1","xywh":"1,1,20,20",
        "section_number":None,
    }
    result = _structure_toc_items(doc, {"items":[raw],"run":None})
    assert len(result) == 1 and result[0]["title"] == raw["title"]
    assert result[0]["source_method"] == "manual"


def test_real_core_tree_orders_pages_and_exposes_stale_model_proposals(tmp_path):
    """Use the actual core view and saved proposals; never load model weights."""
    from src.db.models import AltoBlock, IiifRange
    from src.toc_models import TocNodeProposal, TocStructureRun
    from src.toc_structure import collect_snapshot
    engine = init_db("sqlite:///" + (tmp_path / "library.db").as_posix())
    with get_session(engine) as session:
        work = Work(slug="real-core", title="实际目录契约测试")
        session.add(work); session.flush()
        manifest = Manifest(work_id=work.id, volume_number=1, label="第一卷", total_pages=4)
        session.add(manifest); session.flush()
        titles = ["第一章：绪论", "第一节 研究材料", "第二章 分析", "第一节 分析方法"]
        ranges, blocks = [], []
        for number, title in enumerate(titles, 1):
            canvas = Canvas(
                manifest_id=manifest.id, sequence_index=number, scan_index=number,
                printed_page_label=str(number), width=1200, height=1600,
                iiif_image_service_id=f"test/image/{number}",
            )
            session.add(canvas); session.flush()
            block = AltoBlock(
                canvas_id=canvas.id, alto_element_id=f"h-{number}", block_type="heading",
                page_zone="print_space", reading_order=1, column_index=0,
                x=100, y=100, w=900, h=50, raw_ocr=title, content_text=title,
            )
            session.add(block); session.flush(); blocks.append(block)
            parent = ranges[-1].id if number in {2,4} else None
            row = IiifRange(
                manifest_id=manifest.id, parent_id=parent, level=3 if parent else 2,
                sort_order=1 if parent or number == 1 else 2, title=title,
                source_method="auto_heading", heading_block_id=block.id,
                start_canvas_id=canvas.id, start_label=str(number), start_xywh=block.xywh,
            )
            session.add(row); session.flush(); ranges.append(row)
        frame = collect_snapshot(session, manifest.id)
        run = TocStructureRun(
            manifest_id=manifest.id, request_key="test-real-core", source_digest=frame["source_digest"],
            model_name="test-local.gguf", model_identity={}, config_json={},
            status="completed_with_review",
            summary_json={"processed":1,"total":4,"changes":1,"remaining":3},
        )
        session.add(run); session.flush()
        node = frame["nodes"][0]
        session.add(TocNodeProposal(
            run_id=run.id, node_key=node["key"], source_range_id=ranges[0].id,
            block_id=blocks[0].id, keep=True, level=1, parent_key=None,
            reason="章级标题并有后续节标题", evidence_quote=titles[0],
            source_snapshot={k:node[k] for k in (
                "title","quote","block_id","source_sha256","canvas_id","scan_index","printed_page","xywh"
            )},
        ))
        manifest_id, block_id = manifest.id, blocks[0].id
        range_ids = [r.id for r in ranges]
    server = make_server(tmp_path, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}", timeout=20) as client:
            fresh = client.get("/api/toc_structure", params={"manifest":manifest_id}).json()
            assert fresh["run"]["source_current"] is True
            assert [item["scan_index"] for item in fresh["items"]] == [1,2,3,4]
            data = client.get("/api/toc", params={"manifest":manifest_id,"max_level":0,"source":"local_model"}).json()
            assert [item["id"] for item in data] == range_ids
            assert data[0]["source_method"] == "local_model" and data[0]["review_status"] == "pending"
            assert data[1]["parent_id"] == range_ids[0] and data[3]["parent_id"] == range_ids[2]
            with get_session(engine) as session:
                session.get(AltoBlock, block_id).approved_text = "第一章 修订后的绪论"
            stale = client.get("/api/toc_structure", params={"manifest":manifest_id}).json()
            assert stale["run"]["status"] == "stale" and stale["run"]["source_current"] is False
            assert all(item["source_method"] != "local_model" for item in stale["items"])
    finally:
        server.shutdown(); worker.join(); server.server_close()
        server.engine.dispose(); engine.dispose()


def test_plate_fallback_uses_original_rule_level_not_normalized_depth(tmp_path):
    from src.db.models import AltoBlock, IiifRange, PlateFigure
    from src.server import _build_index_cache
    from src.toc_structure import toc_view
    engine = init_db("sqlite:///" + (tmp_path / "library.db").as_posix())
    try:
        with get_session(engine) as session:
            work = Work(slug="plate-toc", title="第三卷")
            session.add(work); session.flush()
            manifest = Manifest(work_id=work.id, volume_number=3, label="第三卷", total_pages=1)
            session.add(manifest); session.flush()
            canvas = Canvas(
                manifest_id=manifest.id, sequence_index=1, scan_index=1,
                printed_page_label="图版一", page_domain="plate",
                width=1200, height=1600, iiif_image_service_id="test/image",
            )
            session.add(canvas); session.flush()
            caption = "1. 青白釉碗Aa型（T1①∶1）"
            block = AltoBlock(
                canvas_id=canvas.id, alto_element_id="plate-label", block_type="heading",
                page_zone="print_space", reading_order=1, column_index=0,
                x=100,y=900,w=900,h=60,raw_ocr=caption,content_text=caption,
            )
            session.add(block); session.flush()
            session.add(IiifRange(
                manifest_id=manifest.id, parent_id=None, level=6, sort_order=1,
                title=caption, source_method="auto_heading", heading_block_id=block.id,
                start_canvas_id=canvas.id, start_label="图版一", start_xywh=block.xywh,
            ))
            session.add(PlateFigure(
                manifest_id=manifest.id, canvas_id=canvas.id, caption_block_id=block.id,
                figure_category="plate",figure_number="图版2.2.1",caption_text=caption,
                crop_x=100,crop_y=200,crop_w=900,crop_h=500,
            ))
            session.flush()
            baseline = toc_view(session, manifest.id, source="rules")
            assert baseline["items"][0]["level"] == 1
            assert baseline["items"][0]["rule_level"] == 6
            manifest_id = manifest.id
        index = _build_index_cache(engine)
        items = [item for item in index["toc"] if item["manifest_id"] == manifest_id]
        assert items and items[0]["id"].startswith("plate_group_")
        assert all(item["source_method"] == "figure_index" for item in items)
        assert any(item["parent_id"] == items[0]["id"] for item in items[1:])
    finally:
        engine.dispose()
