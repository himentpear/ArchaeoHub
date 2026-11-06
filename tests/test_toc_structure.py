"""Local TOC graph, provenance, cache and failure-isolation regressions."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError
from sqlalchemy import func, select

from src.db.models import AltoBlock, Canvas, IiifRange, Manifest, Work
from src.db.session import get_session, init_db
from src.toc_models import TocNodeProposal, TocStructureRun
from src.toc_structure import (
    _tree, batch_request, collect_snapshot, run_local_toc_review, toc_view, validate_batch,
)


@pytest.fixture
def library(tmp_path):
    engine = init_db("sqlite:///" + (tmp_path / "library.db").as_posix())
    with get_session(engine) as session:
        work = Work(slug="toc-tests", title="本地目录测试报告")
        session.add(work)
        session.flush()
        manifest = Manifest(work_id=work.id, volume_number=1, label="第一卷", total_pages=3)
        session.add(manifest)
        session.flush()
        canvases = []
        for page in range(1, 4):
            canvas = Canvas(
                manifest_id=manifest.id, sequence_index=page, scan_index=page,
                printed_page_label=str(page + 40), page_domain="body",
                width=1200, height=1600, iiif_image_service_id=f"test/image/{page}",
            )
            session.add(canvas)
            session.flush()
            canvases.append(canvas)

        def block(text, page, order, kind="heading", zone="print_space"):
            item = AltoBlock(
                canvas_id=canvases[page - 1].id, alto_element_id=f"b-{page}-{order}",
                block_type=kind, page_zone=zone, reading_order=order,
                column_index=0, x=100, y=100 + order * 100, w=900, h=70,
                content_text=text, raw_ocr=text,
            )
            session.add(item)
            session.flush()
            return item

        def heading(text, page, order, level, parent=None):
            source = block(text, page, order)
            item = IiifRange(
                manifest_id=manifest.id, heading_block_id=source.id,
                start_canvas_id=source.canvas_id, start_label=str(page + 40),
                start_xywh=source.xywh, title=text, level=level,
                parent_id=parent.id if parent else None, sort_order=1,
                source_method="auto_heading",
            )
            session.add(item)
            session.flush()
            return item, source

        chapter, _ = heading("第一章 窑业生产", 1, 1, 1)
        section, _ = heading("第一节 生产遗迹", 1, 2, 2, chapter)
        caption, _ = heading("图1 窑址分布图", 1, 3, 3, section)
        missed = block("生产技术与装烧方法", 2, 1, "paragraph")
        subsection, _ = heading("一、匣钵", 2, 2, 3, section)
        second, _ = heading("第二章 器物分类", 3, 1, 1)
        last, _ = heading("第一节 碗", 3, 2, 2, second)
        excluded = block("第二章 器物分类", 3, 0, "running_header", "top_margin")
        ids = {
            "manifest": manifest.id, "chapter": chapter.id, "section": section.id,
            "caption": caption.id, "subsection": subsection.id, "second": second.id,
            "last": last.id, "missed_block": missed.id, "excluded_block": excluded.id,
            "page2": canvases[1].id,
        }
    yield engine, ids
    engine.dispose()


def frame_for(library):
    engine, ids = library
    with get_session(engine) as session:
        return collect_snapshot(session, ids["manifest"])


def by_title(frame):
    return {node["title"]: node for node in frame["nodes"]}


def decision(node, parent=None, level=1, keep=True):
    return {
        "id": node["key"], "keep": keep, "level": level if keep else 0,
        "parent_id": parent if keep else None, "evidence": node["quote"],
        "reason": "标题、上下文与页序支持此关系" if keep else "此项为图注，不是章节",
    }


def editorial_decisions(frame):
    nodes = by_title(frame)
    chapter = nodes["第一章 窑业生产"]["key"]
    section = nodes["第一节 生产遗迹"]["key"]
    missed = nodes["生产技术与装烧方法"]["key"]
    second = nodes["第二章 器物分类"]["key"]
    return {
        n["key"]: value for n, value in [
            (nodes["第一章 窑业生产"], decision(nodes["第一章 窑业生产"])),
            (nodes["第一节 生产遗迹"], decision(nodes["第一节 生产遗迹"], chapter, 2)),
            (nodes["图1 窑址分布图"], decision(nodes["图1 窑址分布图"], keep=False)),
            (nodes["生产技术与装烧方法"], decision(nodes["生产技术与装烧方法"], section, 3)),
            (nodes["一、匣钵"], decision(nodes["一、匣钵"], missed, 4)),
            (nodes["第二章 器物分类"], decision(nodes["第二章 器物分类"])),
            (nodes["第一节 碗"], decision(nodes["第一节 碗"], second, 2)),
        ]
    }


class FakeModel:
    def __init__(self, values, *, fail_calls=(), on_infer=None):
        self.values = values
        self.fail_calls = set(fail_calls)
        self.on_infer = on_infer
        self.calls = []
        self.closed = False

    def infer(self, request_id, prompt, schema):
        data = json.loads(prompt[prompt.index('{"known_ancestors"'):])
        self.calls.append(data)
        if self.on_infer:
            self.on_infer(len(self.calls))
        if len(self.calls) in self.fail_calls:
            raise RuntimeError("Synthetic local worker failure")
        return {
            "request_id": request_id,
            "result": {"nodes": [copy.deepcopy(self.values[n["id"]]) for n in data["targets"]]},
            "usage": {"total_tokens": 1},
        }

    def close(self):
        self.closed = True


def run_fake(library, tmp_path, client, **kwargs):
    engine, ids = library
    model = tmp_path / "test-model.gguf"
    if not model.exists():
        model.write_bytes(b"synthetic-local-weight")
    return run_local_toc_review(
        engine, ids["manifest"], tmp_path / "run",
        model_path=model, settings_path=tmp_path / "missing-settings.env",
        max_nodes=100, batch_size=kwargs.pop("batch_size", 2),
        max_batches=20, model_client=client, **kwargs,
    )


def test_snapshot_contains_missed_heading_and_keeps_physical_order(library):
    frame = frame_for(library)
    nodes = by_title(frame)
    assert len(frame["nodes"]) == 7 and len(frame["baseline"]) == 6
    assert nodes["生产技术与装烧方法"]["range_id"] is None
    assert nodes["图1 窑址分布图"]["range_id"] is not None
    assert frame["nodes"][0]["scan_index"] == 1
    assert frame["nodes"][-1]["scan_index"] == 3
    assert not any(n["block_id"] == library[1]["excluded_block"] for n in frame["nodes"])
    assert nodes["一、匣钵"]["rule_parent_key"] == nodes["第一节 生产遗迹"]["key"]


def test_tree_excludes_false_caption_adds_missed_title_and_attaches_child(library):
    frame = frame_for(library)
    nodes = by_title(frame)
    parents, levels = _tree(frame, editorial_decisions(frame))
    assert nodes["图1 窑址分布图"]["key"] not in parents
    assert levels[nodes["生产技术与装烧方法"]["key"]] == 3
    assert parents[nodes["一、匣钵"]["key"]] == nodes["生产技术与装烧方法"]["key"]
    assert levels[nodes["一、匣钵"]["key"]] == 4


def test_batch_carries_model_added_cross_page_parent_as_known_ancestor(library):
    frame = frame_for(library)
    nodes = by_title(frame)
    values = editorial_decisions(frame)
    first = frame["nodes"][:4]
    _, _, allowed = batch_request(frame, first, {})
    prior = validate_batch(
        frame, first, {}, {"nodes": [values[n["key"]] for n in first]}, allowed,
    )
    target = [nodes["一、匣钵"]]
    prompt, _, allowed = batch_request(frame, target, prior)
    assert nodes["生产技术与装烧方法"]["key"] in allowed
    assert nodes["生产技术与装烧方法"]["key"] in prompt
    accepted = validate_batch(frame, target, prior, {"nodes": [values[target[0]["key"]]]}, allowed)
    assert accepted[target[0]["key"]]["parent_id"] == nodes["生产技术与装烧方法"]["key"]


@pytest.mark.parametrize("problem", [
    "duplicate", "missing", "unknown_id", "unknown_parent", "future_parent",
    "self_parent", "fabricated_evidence", "wrong_level", "excluded_has_parent",
])
def test_batch_rejects_invalid_model_output(library, problem):
    from src.toc_structure import validate_batch
    frame = frame_for(library)
    targets = frame["nodes"][:2]
    answer = {"nodes": [
        decision(targets[0]), decision(targets[1], targets[0]["key"], 2),
    ]}
    _, _, allowed = batch_request(frame, targets, {})
    if problem == "duplicate":
        answer["nodes"][1] = copy.deepcopy(answer["nodes"][0])
    elif problem == "missing":
        answer["nodes"].pop()
    elif problem == "unknown_id":
        answer["nodes"][0]["id"] = "n99999"
    elif problem == "unknown_parent":
        answer["nodes"][1]["parent_id"] = "n99999"
    elif problem == "future_parent":
        answer["nodes"][0]["parent_id"] = targets[1]["key"]
        answer["nodes"][0]["level"] = 2
    elif problem == "self_parent":
        answer["nodes"][0]["parent_id"] = targets[0]["key"]
    elif problem == "fabricated_evidence":
        answer["nodes"][0]["evidence"] = "第三章 原文没有的标题"
    elif problem == "wrong_level":
        answer["nodes"][1]["level"] = 4
    else:
        answer["nodes"][1]["keep"] = False
    with pytest.raises((ValueError, ValidationError)):
        validate_batch(frame, targets, {}, answer, allowed)


def test_excluded_explicit_parent_is_rejected_even_when_model_lists_child_first(library):
    from src.toc_structure import validate_batch
    frame = frame_for(library)
    targets = frame["nodes"][:2]
    _, _, allowed = batch_request(frame, targets, {})
    answer = {"nodes": [
        decision(targets[1], targets[0]["key"], 1),
        decision(targets[0], keep=False),
    ]}
    with pytest.raises(ValueError):
        validate_batch(frame, targets, {}, answer, allowed)


def test_closed_subtree_cannot_be_reopened_by_a_later_chapter_child(library):
    frame = frame_for(library)
    nodes = by_title(frame)
    proposal = decision(nodes["第一节 碗"], nodes["第一章 窑业生产"]["key"], 2)
    with pytest.raises(ValueError, match="cross"):
        _tree(frame, {proposal["id"]: proposal})


def test_cycles_and_excessive_depth_are_rejected():
    nodes = {str(i): {"order": i, "key": str(i), "locked": False} for i in range(10)}
    frame = {
        "catalog": nodes,
        "baseline": {str(i): {"keep": True, "parent_id": str(i - 1) if i else None} for i in range(10)},
    }
    with pytest.raises(ValueError, match="depth"):
        _tree(frame, {})
    frame["baseline"]["0"]["parent_id"] = "9"
    with pytest.raises(ValueError):
        _tree(frame, {})


def test_full_local_round_is_pending_idempotent_and_does_not_change_rule_ranges(library, tmp_path):
    engine, ids = library
    frame = frame_for(library)
    values = editorial_decisions(frame)
    with get_session(engine) as session:
        baseline = [(r.id, r.parent_id, r.level, r.title) for r in session.scalars(select(IiifRange))]
    client = FakeModel(values)
    result = run_fake(library, tmp_path, client)
    assert result["processed"] == 7 and result["status"] == "completed"
    assert result["added"] == result["excluded"] == result["reparented"] == 1
    assert result["cloud_calls"] == 0 and client.closed
    assert any(n["id"] == by_title(frame)["生产技术与装烧方法"]["key"]
               for call in client.calls[2:] for n in call["known_ancestors"])
    with get_session(engine) as session:
        after = [(r.id, r.parent_id, r.level, r.title) for r in session.scalars(select(IiifRange))]
        assert after == baseline
        model_view = toc_view(session, ids["manifest"], source="local_model")
        titles = {n["title"]: n for n in model_view["items"]}
        assert "图1 窑址分布图" not in titles
        assert titles["一、匣钵"]["parent_id"] == titles["生产技术与装烧方法"]["id"]
        assert titles["一、匣钵"]["review_status"] == "pending"
        rules = toc_view(session, ids["manifest"], source="rules")
        assert "图1 窑址分布图" in {n["title"] for n in rules["items"]}
        assert "生产技术与装烧方法" not in {n["title"] for n in rules["items"]}
        before_proposals = session.scalar(select(func.count()).select_from(TocNodeProposal))
    cached_client = FakeModel(values)
    cached = run_fake(library, tmp_path, cached_client)
    assert cached["cached"] and not cached_client.calls
    with get_session(engine) as session:
        assert session.scalar(select(func.count()).select_from(TocStructureRun)) == 1
        assert session.scalar(select(func.count()).select_from(TocNodeProposal)) == before_proposals


def test_partial_model_failure_retains_unprocessed_baseline(library, tmp_path):
    engine, ids = library
    client = FakeModel(editorial_decisions(frame_for(library)), fail_calls={4})
    result = run_fake(library, tmp_path, client)
    assert result["status"] == "completed_with_review"
    assert result["processed"] == 6 and result["remaining"] == 1 and len(result["errors"]) == 1
    with get_session(engine) as session:
        view = toc_view(session, ids["manifest"], source="local_model")
        titles = {n["title"]: n for n in view["items"]}
        assert titles["第一节 碗"]["id"] == ids["last"]
        assert titles["第一节 碗"]["source_method"] == "auto_heading"
        assert titles["第一节 碗"]["parent_id"] == ids["second"]
        assert session.scalar(select(func.count()).select_from(IiifRange)) == 6


def test_manual_source_is_locked_and_preserved_in_rule_view(library):
    engine, ids = library
    with get_session(engine) as session:
        session.get(IiifRange, ids["section"]).source_method = "manual"
    frame = frame_for(library)
    node = by_title(frame)["第一节 生产遗迹"]
    assert node["locked"]
    from src.toc_structure import validate_batch
    _, _, allowed = batch_request(frame, [node], {})
    with pytest.raises(ValueError, match="Manually"):
        validate_batch(frame, [node], {}, {"nodes": [decision(node, node["rule_parent_key"], 2)]}, allowed)
    with get_session(engine) as session:
        item = next(i for i in toc_view(session, ids["manifest"], source="rules")["items"] if i["id"] == ids["section"])
        assert item["source_method"] == "manual"


def test_excluding_ancestor_cannot_implicitly_reparent_manually_reviewed_node(library):
    engine, ids = library
    with get_session(engine) as session:
        session.get(IiifRange, ids["section"]).source_method = "manual"
    frame = frame_for(library)
    chapter = by_title(frame)["第一章 窑业生产"]
    with pytest.raises(ValueError):
        _tree(frame, {chapter["key"]: decision(chapter, keep=False)})


@pytest.mark.parametrize("change", ["text", "page_domain", "printed_page"])
def test_source_change_invalidates_model_view(library, tmp_path, change):
    engine, ids = library
    frame = frame_for(library)
    run_fake(library, tmp_path, FakeModel(editorial_decisions(frame)))
    with get_session(engine) as session:
        if change == "text":
            session.get(AltoBlock, ids["missed_block"]).approved_text = "修订后的生产技术标题"
        elif change == "page_domain":
            session.get(Canvas, ids["page2"]).page_domain = "plate"
        else:
            session.get(Canvas, ids["page2"]).printed_page_label = "修订页码"
    with get_session(engine) as session:
        view = toc_view(session, ids["manifest"])
        assert view["run"]["source_current"] is False
        assert view["run"]["status"] == "stale"
        assert not any(n["source_method"] == "local_model" for n in view["items"])


def test_source_edit_during_inference_prevents_commit(library, tmp_path):
    engine, ids = library
    def edit(call):
        if call == 1:
            with get_session(engine) as session:
                session.get(AltoBlock, ids["missed_block"]).approved_text = "外部同时校订的标题"
    client = FakeModel(editorial_decisions(frame_for(library)), on_infer=edit)
    result = run_fake(library, tmp_path, client)
    assert result["status"] == "failed" and result["processed"] == 0
    assert result["errors"]
    with get_session(engine) as session:
        assert session.scalar(select(func.count()).select_from(TocNodeProposal)) == 0
        assert session.scalar(select(func.count()).select_from(IiifRange)) == 6


def test_modified_source_does_not_reuse_completed_run(library, tmp_path):
    engine, ids = library
    first = run_fake(library, tmp_path, FakeModel(editorial_decisions(frame_for(library))))
    with get_session(engine) as session:
        source = session.get(AltoBlock, ids["missed_block"])
        source.approved_text = source.content_text + "\n新增的版本依据"
    client = FakeModel(editorial_decisions(frame_for(library)))
    second = run_fake(library, tmp_path, client)
    assert second["id"] != first["id"]
    assert not second.get("cached") and client.calls
    with get_session(engine) as session:
        assert session.scalar(select(func.count()).select_from(TocStructureRun)) == 2
        assert toc_view(session, ids["manifest"])["run"]["source_current"]


def test_timeout_preserves_valid_earlier_batches_and_closes_local_worker(library, tmp_path):
    def timeout(call):
        if call == 2:
            raise TimeoutError("Synthetic local timeout")
    client = FakeModel(editorial_decisions(frame_for(library)), on_infer=timeout)
    result = run_fake(library, tmp_path, client)
    assert result["status"] == "completed_with_review" and result["processed"] == 2
    assert result["remaining"] == 5 and len(result["errors"]) == 1
    assert client.closed
    with get_session(library[0]) as session:
        view = toc_view(session, library[1]["manifest"])
        assert len(view["items"]) == 6
        assert "第一节 碗" in {item["title"] for item in view["items"]}


def test_missing_local_model_never_falls_back_to_cloud(library, tmp_path, monkeypatch):
    from src.model_router import ModelRouter
    def forbidden(*args, **kwargs):
        pytest.fail("Local TOC review must never invoke a cloud endpoint")
    monkeypatch.setattr(ModelRouter, "_cloud", forbidden)
    monkeypatch.setenv("ALLOW_CLOUD_FALLBACK", "true")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-fake-secret")
    engine, ids = library
    result = run_local_toc_review(
        engine, ids["manifest"], tmp_path / "missing-model-run",
        model_path=tmp_path / "missing.gguf", settings_path=tmp_path / "missing.env",
        max_nodes=10, batch_size=2, max_batches=5,
    )
    assert result["status"] == "unavailable" and result["cloud_calls"] == 0
    assert "test-only-fake-secret" not in (tmp_path / "missing-model-run" / "toc_model_stage.json").read_text()


def test_edge_only_decisions_get_exact_source_evidence_and_computed_levels(library):
    from src.toc_structure import hydrate_local_decisions, local_decision_request
    frame = frame_for(library)
    targets = frame["nodes"][:3]
    _, schema, allowed = local_decision_request(frame, targets, {})
    values = editorial_decisions(frame)
    raw = {"nodes": [{
        "id": n["key"], "keep": values[n["key"]]["keep"],
        "parent_id": values[n["key"]]["parent_id"], "reason": "根据标题类型与上下文判断",
    } for n in targets]}
    # Parent/level on an excluded item has no semantic role. Only the explicit
    # exclusion is applied; no structure is created from its old parent.
    raw["nodes"][2]["parent_id"] = targets[0]["key"]
    result = hydrate_local_decisions(frame, targets, {}, raw, schema, allowed)
    assert result[targets[0]["key"]]["level"] == 1
    assert result[targets[1]["key"]]["level"] == 2
    assert result[targets[2]["key"]]["level"] == 0
    assert result[targets[2]["key"]]["parent_id"] is None
    assert all(result[n["key"]]["evidence"] == n["quote"] for n in targets)
    assert all("evidence" not in value for value in raw["nodes"])


def test_real_worker_output_contract_and_iiif_share_the_proposed_tree(library, tmp_path):
    from src.iiif.generator import IiifBuilder
    frame = frame_for(library)
    values = editorial_decisions(frame)
    class EdgeOnlyModel(FakeModel):
        def infer(self, request_id, prompt, schema):
            result = super().infer(request_id, prompt, schema)
            result["result"]["nodes"] = [{key: value[key] for key in ("id", "keep", "parent_id", "reason")}
                                          for value in result["result"]["nodes"]]
            return result
    result = run_fake(library, tmp_path, EdgeOnlyModel(values))
    assert result["processed"] == 7 and not result["errors"]
    with get_session(library[0]) as session:
        view = toc_view(session, library[1]["manifest"], source="local_model")
        manifest = IiifBuilder(session).build_manifest(library[1]["manifest"], toc_source="local_model")
        flattened = []
        def walk(ranges, parent=None):
            for item in ranges:
                if item["type"] != "Range":
                    continue
                identifier = item["id"].rsplit("/", 1)[-1]
                flattened.append((identifier, parent))
                walk(item["items"], identifier)
        walk(manifest["structures"])
        assert flattened == [(node["id"], node["parent_id"]) for node in view["items"]]
        assert "待人工核验" in json.dumps(manifest, ensure_ascii=False)


def test_pending_model_judgments_do_not_replace_default_directory(library, tmp_path):
    frame = frame_for(library)
    run_fake(library, tmp_path, FakeModel(editorial_decisions(frame)))
    with get_session(library[0]) as session:
        default = toc_view(session, library[1]["manifest"])
        rules = toc_view(session, library[1]["manifest"], source="rules")
        preview = toc_view(session, library[1]["manifest"], source="local_model")
        assert default["items"] == rules["items"]
        assert all(n["source_method"] != "local_model" for n in default["items"])
        assert default["run"]["suggestions"]
        assert any(n["source_method"] == "local_model" for n in preview["items"])
        assert "图1 窑址分布图" in {n["title"] for n in default["items"]}
        assert "图1 窑址分布图" not in {n["title"] for n in preview["items"]}
