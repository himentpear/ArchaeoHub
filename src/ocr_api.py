"""PaddleOCR-VL document recognition through an independent OpenAI-compatible API."""
import base64
import json
import math
import mimetypes
from pathlib import Path

from src.io_utils import atomic_write
from src.model_api import OpenAIEndpoint
from src.settings import load_settings

OCR_MODEL = "PaddleOCR-VL-1.6-0.9B"


def save_ocr_result(result, directory):
    directory=Path(directory)
    for relative,encoded in result.get('assets',{}).items():
        atomic_write(directory/relative,base64.b64decode(encoded,validate=True))
    atomic_write(directory/'result.json',(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf8'))
    atomic_write(directory/'document.md',result['markdown'].encode('utf8'))


class PaddleOcrClient:
    def __init__(self, settings=None):
        settings = load_settings() if settings is None else settings
        self.endpoint = OpenAIEndpoint(settings.get("OCR_BASE_URL", "http://127.0.0.1:18082/v1"),
            settings.get("OCR_MODEL", OCR_MODEL), settings.get("OCR_API_KEY"),
            timeout=float(settings.get("OCR_TIMEOUT", "1800")))
        self.max_tokens = int(settings.get("OCR_MAX_TOKENS", "8192"))
        self.max_pixels = int(settings.get("OCR_MAX_PIXELS", "1003520"))
        if not 16 <= self.max_tokens <= 16384 or not 16384 <= self.max_pixels <= 4000000:
            raise ValueError("Invalid PaddleOCR-VL generation limits")

    @property
    def identity(self):
        return {"model": self.endpoint.model, "endpoint": self.endpoint.base_url,
                "max_tokens": self.max_tokens, "max_pixels": self.max_pixels, "adapter_version": 1}

    def recognize(self, image, *, task="ocr", output_dir=None, max_tokens=None):
        if task not in {"ocr", "text", "table", "spotting"}:
            raise ValueError("Unsupported OCR task")
        limit=self.max_tokens if max_tokens is None else max_tokens
        if type(limit) is not int or not 16<=limit<=16384:
            raise ValueError('Invalid OCR token limit')
        if isinstance(image, bytes):
            data, mime = image, "image/png"
        else:
            image = Path(image)
            data, mime = image.read_bytes(), mimetypes.guess_type(image.name)[0] or "image/png"
        content = [{"type": "text", "text": "Recognize this document image."},
                   {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + base64.b64encode(data).decode()}}]
        response = self.endpoint.complete([{"role": "user", "content": content}], max_tokens=limit,
            extra={"metadata": {"ocr_task": task, "max_pixels": str(self.max_pixels)}})
        result = json.loads(response["choices"][0]["message"]["content"])
        self.validate(result)
        if output_dir is not None:
            save_ocr_result(result,output_dir)
        return result

    def validate(self, result):
        if not isinstance(result, dict) or result.get("model") != self.endpoint.model:
            raise ValueError("Unexpected OCR model response")
        size = result.get("image_size")
        if (not isinstance(size, list) or len(size) != 2
                or any(type(value) is not int or value < 1 for value in size)):
            raise ValueError("Invalid OCR image dimensions")
        if not isinstance(result.get("blocks"), list) or not isinstance(result.get("markdown"), str):
            raise ValueError("Invalid OCR document result")
        assets = result.get("assets", {})
        if not isinstance(assets, dict):
            raise ValueError("Invalid Markdown image assets")
        for relative, encoded in assets.items():
            if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                    or ".." in Path(relative).parts or Path(relative).drive or not isinstance(encoded, str)):
                raise ValueError("Invalid Markdown image path")
            base64.b64decode(encoded, validate=True)
        for block in result["blocks"]:
            if not isinstance(block, dict) or not isinstance(block.get("text"), str) or not isinstance(block.get("label"), str):
                raise ValueError("Invalid OCR block")
            box = block.get("bbox")
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(value) not in {int, float} or not math.isfinite(value) for value in box)
                    or not 0 <= box[0] < box[2] <= size[0] or not 0 <= box[1] < box[3] <= size[1]):
                raise ValueError("OCR block is outside its source image")
