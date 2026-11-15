"""Resumable PDF extraction using native text and an independent PaddleOCR-VL API."""
from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from src.ingest.digital_pdf_ingestor import DigitalPdfIngestor, ExtractedLine, ExtractedPageData
from src.ingest.layout import order_lines
from src.io_utils import atomic_write

PIPELINE_VERSION = "4.0.0"


def atomic_json(path: Path, value: object) -> None:
    atomic_write(path,(json.dumps(value,ensure_ascii=False,indent=2)+"\n").encode("utf-8"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refine_graphic_page(page: ExtractedPageData) -> ExtractedPageData:
    area=page.width_pt*page.height_pt
    graphic_area=sum(r[2]*r[3] for r in page.graphics)
    if (page.layout_hint not in {"cover","toc"} and len(page.graphics)>=2
            and graphic_area>area*0.2 and sum(len(l.text) for l in page.lines)<700):
        page.layout_hint="plate"
    return page


@contextmanager
def job_lock(path: Path):
    """OS-released advisory lock, including when the process is killed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class LocalOcr:
    def __init__(self, threads: int = 4, settings=None):
        from src.ocr_api import PaddleOcrClient
        self.client = PaddleOcrClient(settings)

    def recognize(self, image_path: Path, scale: float) -> list[ExtractedLine]:
        from src.graphics.vision import parse_structure
        result = self.client.recognize(image_path, output_dir=Path(image_path).parent.parent / "ocr-vl" / Path(image_path).stem)
        lines = []
        self.regions = []
        for block in result["blocks"]:
            text = block["text"]
            if block["label"] == "table":
                rows = {}
                for cell in parse_structure(text):
                    rows.setdefault(cell["row"], []).append(cell["text"])
                text = "\n".join("\t".join(cells) for cells in rows.values())
            x0, y0, x1, y1 = [value / scale for value in block["bbox"]]
            if block["label"] in {"table", "image", "chart"}:
                self.regions.append({"category": block["label"], "key": "vl-" + str(len(self.regions)),
                                     "bbox_pt": [x0, y0, x1, y1]})
            if text.strip():
                lines.append(ExtractedLine(text, x0, y0, x1-x0, y1-y0, min(14., (y1-y0)*.75), None))
        return order_lines(lines)


def native_quality(page, lines: list[ExtractedLine]) -> tuple[str, list[str]]:
    text = "".join(l.text for l in lines)
    if text and (text.count("\ufffd")/len(text) > 0.01 or
                 sum(ord(c) < 32 and not c.isspace() for c in text)/len(text) > 0.01):
        return "ocr", ["damaged_text_layer"]
    # An image covering most of a page can contain unrecognized text even if a
    # footer or a small part of a previous OCR layer remains selectable.
    area = page.rect.width*page.rect.height
    import pymupdf
    image_area = max((pymupdf.Rect(i["bbox"]).get_area() for i in page.get_image_info()), default=0)
    if image_area > area*0.65 and len(text) < 80:
        return "ocr", ["full_page_image_with_sparse_text"]
    if not text and image_area > area*0.02:
        return "ocr", ["image_without_text"]
    if not text:
        # Vector outlines may represent text; a render is needed to tell from blank.
        pix = page.get_pixmap(matrix=pymupdf.Matrix(0.3, 0.3), colorspace=pymupdf.csGRAY)
        import numpy as np
        ink = float((np.frombuffer(pix.samples, dtype=np.uint8)<220).mean())
        return ("blank", []) if ink < 0.0003 else ("ocr", ["visible_content_without_text"])
    return "native", []


class PageExtractor:
    def __init__(self, run_dir: Path, dpi: int = 180, threads: int = 4, graphics_front_dir: Path | None = None,
                 settings=None):
        self.run_dir = run_dir
        self.dpi = dpi
        self.threads = threads
        self.ocr = None
        self.graphics_front_dir=graphics_front_dir
        from src.ocr_api import PaddleOcrClient
        self.ocr_settings = settings
        self.ocr_identity = PaddleOcrClient(settings).identity

    def attach_regions(self,page):
        if self.graphics_front_dir is None:return page
        path=Path(self.graphics_front_dir)/f'{page.scan_index:05d}.json'
        data=json.loads(path.read_text(encoding='utf8'))
        if data['signature']['source_hash']!=file_hash(Path(page.image_path)):
            raise ValueError('Graphic layout belongs to a different rendered source page')
        sx=page.width_pt/data['image_size'][0];sy=page.height_pt/data['image_size'][1]
        for region in data['regions']:
            box=[region['bbox'][0]*sx,region['bbox'][1]*sy,region['bbox'][2]*sx,region['bbox'][3]*sy]
            if not any(item['category']==region['category'] and item['bbox_pt']==box for item in page.regions):
                page.regions.append({'category':region['category'],'key':region['key'],'bbox_pt':box})
        return page

    def extract(self, document, index: int, source_sha256: str) -> ExtractedPageData:
        page = document[index]
        cache = self.run_dir / "pages" / f"{index+1:05d}.json"
        image = self.run_dir / "images" / f"{index+1:05d}.jpg"
        signature = {"version": PIPELINE_VERSION, "sha256": source_sha256, "dpi": self.dpi, "ocr": self.ocr_identity}
        if cache.exists() and image.exists():
            try:
                saved = json.loads(cache.read_text(encoding="utf-8"))
                if saved["signature"] == signature and saved["image_sha256"] == file_hash(image):
                    payload = saved["page"]
                    payload["lines"] = order_lines([ExtractedLine(**line) for line in payload["lines"]])
                    # Layout rules may be refined without rerunning OCR. The cached
                    # source lines remain authoritative; derive hints again.
                    payload["detected_page_no_raw"], payload["layout_hint"] = DigitalPdfIngestor._detect_page_number_and_domain_hint(
                        index, payload["lines"], payload["height_pt"])
                    return self.attach_regions(refine_graphic_page(ExtractedPageData(**payload)))
            except (KeyError, ValueError, TypeError):
                pass
        import pymupdf
        started = time.perf_counter()
        image.parent.mkdir(parents=True, exist_ok=True)
        scale = self.dpi / 72
        page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).save(image, jpg_quality=88)
        lines = DigitalPdfIngestor.extract_native_lines(page)
        route, issues = native_quality(page, lines)
        ocr_regions = []
        if route == "ocr":
            if self.ocr is None:
                self.ocr = LocalOcr(self.threads, self.ocr_settings)
            lines = self.ocr.recognize(image, scale)
            ocr_regions = self.ocr.regions
            issues.append("ocr_geometry_is_block_level")
            if not lines:
                issues.append("ocr_no_text_review_image")
            method = "PaddleOCR-VL-v1.6"
        else:
            method = "pymupdf" if route == "native" else "blank"
        # Normal page/figure images are useful graphic regions. A full-page scan
        # is not itself a figure: never use its container rectangle as a figure.
        graphics = []
        for info in page.get_image_info():
            rect = pymupdf.Rect(info["bbox"]) * page.rotation_matrix
            rect &= page.rect
            ratio=rect.get_area()/page.rect.get_area()
            scan_container=route=='ocr' and ratio>=.95
            if not rect.is_empty and ratio>0.01 and not scan_container:
                graphics.append([rect.x0, rect.y0, rect.width, rect.height])
        label, hint = DigitalPdfIngestor._detect_page_number_and_domain_hint(index, lines, page.rect.height)
        result = ExtractedPageData(index+1, index, page.get_label(), page.rect.width,
                                   page.rect.height, len(page.get_images()), lines,
                                   label, hint, method, str(image.resolve()), graphics, issues, ocr_regions)
        result=refine_graphic_page(result)
        result=self.attach_regions(result)
        atomic_json(cache, {"signature": signature, "page": asdict(result), "route": route,
                            "image_sha256": file_hash(image), "elapsed_seconds": round(time.perf_counter()-started,3)})
        return result
