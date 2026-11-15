"""Run PaddleOCRVL(pipeline_version='v1.6') as a local OpenAI-compatible API."""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / "data/processed/models/graphics-weights"))
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/models/ocr-vl-cache")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, str(args.threads))
    from paddleocr import PaddleOCRVL
    import paddle
    from paddlex.inference import load_pipeline_config
    from src.ocr_api import OCR_MODEL
    from src.ocr_server import PaddleOcrService, make_ocr_server

    config = load_pipeline_config("PaddleOCR-VL-1.6")
    config["batch_size"] = 1
    config["SubModules"]["VLRecognition"]["batch_size"] = 1
    pipeline = PaddleOCRVL(pipeline_version="v1.6", device=args.device, cpu_threads=args.threads,
                          enable_mkldnn=False, use_queues=False, paddlex_config=config)
    identity = {"model": OCR_MODEL, "pipeline_version": "v1.6", "device": args.device,
                "paddleocr_version": importlib.metadata.version("paddleocr"),
                "paddle_version": paddle.__version__}
    for field, relative in {"weights_sha256": "PaddleOCR-VL-1.6/model.safetensors",
                            "layout_weights_sha256": "PP-DocLayoutV3/inference.pdiparams"}.items():
        weights = Path(os.environ["PADDLE_PDX_CACHE_HOME"]) / "official_models" / relative
        with weights.open("rb") as stream:
            identity[field] = hashlib.file_digest(stream, "sha256").hexdigest()
    service = PaddleOcrService(pipeline, args.output, identity)
    server = make_ocr_server(service, port=args.port)
    print(f"PaddleOCR-VL v1.6 ready at http://127.0.0.1:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        pipeline.close()


if __name__ == "__main__":
    main()
