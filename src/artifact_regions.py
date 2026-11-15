"""Validated, versioned specimen crops over original page images."""
import io
from pathlib import Path

from PIL import Image
from sqlalchemy import select, update

from src.db.models import Artifact, Canvas, Manifest, Revision
from src.ingest.page_pipeline import file_hash
from src.region_models import ArtifactImageRegion


def image_path(canvas, roots):
    path = Path(canvas.image_storage_path or "").resolve()
    if not path.is_file() or not any(path.is_relative_to(Path(root).resolve()) for root in roots):
        raise LookupError("Source image not available in this library")
    return path


def region_state(region):
    return {"id": region.id, "artifact_id": region.artifact_id, "canvas_id": region.canvas_id,
            "xywh": region.xywh, "view_label": region.view_label, "media_type": region.media_type,
            "status": region.status, "revision": region.revision_no, "image_hash": region.image_hash,
            "source_width": region.source_width, "source_height": region.source_height,
            "image_url": "/api/artifact_region_image?id=" + region.id}


def region_current(region, canvas, roots):
    try:
        return (canvas.width == region.source_width and canvas.height == region.source_height
                and file_hash(image_path(canvas, roots)) == region.image_hash)
    except LookupError:
        return False


def list_regions(session, *, artifact_id=None, canvas_id=None):
    stmt = select(ArtifactImageRegion)
    if artifact_id:
        stmt = stmt.where(ArtifactImageRegion.artifact_id == artifact_id)
    if canvas_id:
        stmt = stmt.where(ArtifactImageRegion.canvas_id == canvas_id)
    return [region_state(region) for region in session.scalars(stmt.order_by(ArtifactImageRegion.id))]


def save_region(session, roots, *, artifact_id, canvas_id, xywh, view_label, media_type,
                actor, note, expected_image_hash, expected_revision=0, region_id=None, status="approved"):
    if not isinstance(actor, str) or not actor.strip() or len(actor) > 128:
        raise ValueError("填写审校人，长度上限为128字")
    if not isinstance(note, str) or not note.strip() or len(note) > 255:
        raise ValueError("填写裁切依据，长度上限为255字")
    if not isinstance(view_label, str) or len(view_label) > 128:
        raise ValueError("视图名称长度上限为128字")
    if media_type not in {"line_drawing", "photo"} or status not in {"approved", "rejected"}:
        raise ValueError("Invalid region type or status")
    artifact, canvas = session.get(Artifact, artifact_id), session.get(Canvas, canvas_id)
    if artifact is None or canvas is None:
        raise LookupError("Artifact or page not found")
    from src.artifact_identity import work_namespace
    source_work = session.get(Manifest, canvas.manifest_id).work
    if work_namespace(artifact.work) != work_namespace(source_work):
        raise ValueError("裁片来源与器物所属报告体系不一致")
    if (not isinstance(xywh, list) or len(xywh) != 4
            or any(type(value) is not int for value in xywh)):
        raise ValueError("裁框应为四个整数 x、y、宽、高")
    x, y, width, height = xywh
    if x < 0 or y < 0 or width < 2 or height < 2 or x + width > canvas.width or y + height > canvas.height:
        raise ValueError("裁框超出原页范围或尺寸过小")
    source_hash = file_hash(image_path(canvas, roots))
    if source_hash != expected_image_hash:
        raise ValueError("原页图像已更新，请重新打开后裁切")
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("Invalid crop revision")
    before = {}
    values = {"xywh": xywh, "view_label": view_label.strip(), "media_type": media_type,
              "status": status, "image_hash": source_hash, "source_width": canvas.width,
              "source_height": canvas.height, "revision_no": expected_revision + 1}
    if region_id:
        region = session.get(ArtifactImageRegion, region_id)
        if region is None:
            raise LookupError("Region not found")
        if region.artifact_id != artifact_id or region.canvas_id != canvas_id:
            raise ValueError("裁片与器物或原页不一致")
        before = region_state(region)
        claimed = session.execute(update(ArtifactImageRegion).where(
            ArtifactImageRegion.id == region_id, ArtifactImageRegion.revision_no == expected_revision,
        ).values(**values).execution_options(synchronize_session=False))
        if claimed.rowcount != 1:
            raise ValueError("裁片版本已更新，请重新打开后修改")
        session.refresh(region)
    else:
        if expected_revision != 0:
            raise ValueError("新裁片的初始版本应为0")
        region = ArtifactImageRegion(artifact_id=artifact_id, canvas_id=canvas_id, **values)
        session.add(region)
        session.flush()
    after = region_state(region)
    session.add(Revision(entity_type="artifact_image_region", entity_id=region.id,
                         revision_no=region.revision_no, actor=actor.strip(), reason=note.strip(),
                         before_state=before, after_state=after))
    session.flush()
    return after


def render_region(session, region_id, roots):
    region = session.get(ArtifactImageRegion, region_id)
    if region is None or region.status != "approved":
        raise LookupError("Region not found")
    canvas = session.get(Canvas, region.canvas_id)
    if not region_current(region, canvas, roots):
        raise ValueError("Source image changed; review the crop again")
    with Image.open(image_path(canvas, roots)) as source:
        image = source.convert("RGB").resize((canvas.width, canvas.height))
    x, y, width, height = region.xywh
    output = io.BytesIO()
    image.crop((x, y, x + width, y + height)).save(output, "JPEG", quality=95)
    return output.getvalue()
