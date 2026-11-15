"""Serve the PaddleOCR-VL v1.6 pipeline without loading it into the reader."""
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time

from PIL import Image, UnidentifiedImageError

from src.io_utils import atomic_write
from src.ocr_api import OCR_MODEL


class PaddleOcrService:
    def __init__(self, pipeline, directory, identity):
        self.pipeline, self.directory, self.identity = pipeline, Path(directory), identity
        self.lock = threading.Lock()

    def recognize(self, data, task, max_tokens, max_pixels):
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image.verify()
        signature = {**self.identity, "task": task, "max_tokens": max_tokens, "max_pixels": max_pixels,
                     "source_hash": hashlib.sha256(data).hexdigest()}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        directory = self.directory / key
        cache = directory / "result.json"
        with self.lock:
            if cache.is_file():
                return json.loads(cache.read_text(encoding="utf-8"))
            directory.mkdir(parents=True, exist_ok=True)
            source = directory / "source.png"
            with Image.open(io.BytesIO(data)) as image:
                image.convert("RGB").save(source)
            options = {"max_new_tokens": max_tokens, "max_pixels": max_pixels}
            if task != "ocr":
                options.update(use_layout_detection=False, prompt_label=task if task in {'table','spotting'} else 'ocr')
            started = time.perf_counter()
            results = self.pipeline.predict(str(source), **options)
            if len(results) != 1:
                raise ValueError("Expected one OCR result for one image")
            result = results[0]
            result.save_to_json(save_path=str(directory / "native"))
            result.save_to_markdown(save_path=str(directory / "native"))
            assets = {}
            for asset in (directory / "native").rglob("*"):
                if asset.is_file() and asset.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    assets[asset.relative_to(directory / "native").as_posix()] = base64.b64encode(asset.read_bytes()).decode()
            raw = result.json
            if isinstance(raw, str):
                raw = json.loads(raw)
            raw = raw.get("res", raw)
            blocks = []
            source_blocks=raw.get('parsing_res_list',[])
            if task=='spotting':
                source_blocks=[]
                spotting=raw.get('spotting_res',{})
                entries=spotting if isinstance(spotting,list) else [spotting]
                for entry in entries:
                    for text,polygon in zip(entry.get('rec_texts',[]),entry.get('rec_polys',[])):
                        if not text.strip() or not polygon:continue
                        source_blocks.append({'block_label':'textline','block_content':text,
                            'block_bbox':[min(point[0] for point in polygon),min(point[1] for point in polygon),
                                          max(point[0] for point in polygon),max(point[1] for point in polygon)]})
            for block in source_blocks:
                x0, y0, x1, y1 = block["block_bbox"]
                box = [max(0, float(x0)), max(0, float(y0)), min(width, float(x1)), min(height, float(y1))]
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError("Invalid PaddleOCR-VL source geometry")
                blocks.append({"label": block["block_label"], "text": block["block_content"], "bbox": box})
            payload = {"model": OCR_MODEL, "pipeline_version": "v1.6", "image_size": [width, height],
                       "blocks": blocks, "markdown": result.markdown["markdown_texts"], "assets": assets, "raw_result": raw,
                       "signature": signature, "seconds": round(time.perf_counter() - started, 3),
                       "geometry_level": "line" if task=='spotting' else "block", "requires_review": True}
            atomic_write(cache, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode())
            print(json.dumps({"stage": "ocr", "task": task, "blocks": len(blocks), "seconds": payload["seconds"]}), flush=True)
            return payload


def make_ocr_server(service, port=18082):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self.send({"status": "ok", **service.identity})
            if self.path == "/v1/models":
                return self.send({"object": "list", "data": [{"id": OCR_MODEL, "object": "model", **service.identity}]})
            self.send({"error": {"message": "Not found"}}, 404)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.send({"error": {"message": "Not found"}}, 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 64000000:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("JSON object required")
                if request.get("model") != OCR_MODEL:
                    return self.send({"error": {"message": "Requested OCR model is not served"}}, 404)
                messages = request.get("messages")
                if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
                    raise ValueError("Invalid messages")
                images = [part for message in messages if isinstance(message.get("content"), list)
                          for part in message["content"] if isinstance(part, dict) and part.get("type") == "image_url"]
                if len(images) != 1 or not isinstance(images[0].get("image_url"), dict):
                    raise ValueError("Provide one document image")
                url = images[0]["image_url"].get("url", "")
                if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                    raise ValueError("Provide a base64 image_url")
                data = base64.b64decode(url.split(";base64,", 1)[1], validate=True)
                metadata = request.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise ValueError("Invalid OCR metadata")
                task = metadata.get("ocr_task", "ocr")
                max_tokens, max_pixels = int(request.get("max_tokens", 8192)), int(metadata.get("max_pixels", 1003520))
                if task not in {"ocr", "text", "table", "spotting"} or not 16 <= max_tokens <= 16384 or not 16384 <= max_pixels <= 4000000:
                    raise ValueError("Invalid OCR task or generation limits")
                result = service.recognize(data, task, max_tokens, max_pixels)
                self.send({"id": "ocr-" + result["signature"]["source_hash"][:24], "object": "chat.completion",
                           "created": int(time.time()), "model": OCR_MODEL, "choices": [{"index": 0,
                           "message": {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
                           "finish_reason": "stop"}]})
            except (ValueError, TypeError, KeyError, UnidentifiedImageError) as error:
                self.send({"error": {"message": str(error), "type": "invalid_request_error"}}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                self.send({"error": {"message": "PaddleOCR-VL inference failed", "type": type(error).__name__}}, 500)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
