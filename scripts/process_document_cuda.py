"""CUDA entry point using a GPU-hosted PaddleOCR-VL service."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ocr_api import PaddleOcrClient
from src.settings import load_settings

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--settings', type=Path)
options, _ = parser.parse_known_args()
client = PaddleOcrClient(load_settings(options.settings))
headers = {'Authorization': 'Bearer ' + client.endpoint.api_key} if client.endpoint.api_key else {}
response = httpx.get(client.endpoint.base_url.removesuffix('/v1') + '/health', headers=headers, timeout=20, trust_env=False)
response.raise_for_status()
identity = response.json()
if not identity.get('device', '').startswith('gpu'):
    raise SystemExit('The CUDA entry point requires the OCR service to run with --device gpu')
started = time.perf_counter()
runpy.run_path(str(ROOT / 'scripts/process_document.py'), run_name='__main__')
report = {**identity, 'seconds': round(time.perf_counter() - started, 3), 'ocr': client.identity}
output = ROOT / 'data/processed/gpu-evidence/last-run.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2), encoding='utf-8')
