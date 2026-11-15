"""Compatibility stdio adapter to the independently deployed model API."""
import argparse
import json
import sys
from pathlib import Path

from src.model_api import OpenAIEndpoint
from src.settings import load_settings


def main():
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1200)
    args = parser.parse_args()
    settings = load_settings()
    endpoint = OpenAIEndpoint(
        settings.get("LOCAL_TOC_BASE_URL") or settings.get("LOCAL_LLM_BASE_URL") or "http://127.0.0.1:18081/v1",
        Path(args.model).name,
        settings.get("LOCAL_TOC_API_KEY") or settings.get("LOCAL_LLM_API_KEY"),
        settings.get("LOCAL_TOC_RESPONSE_FORMAT") or settings.get("LOCAL_LLM_RESPONSE_FORMAT", "json_schema"),
        timeout=float(settings.get("LOCAL_TOC_BATCH_TIMEOUT", "150")),
    )
    try:
        for line in sys.stdin:
            request = json.loads(line)
            try:
                result = endpoint.complete([{"role": "user", "content": request["prompt"]}],
                                           schema=request["schema"], max_tokens=args.max_tokens)
                choice = result["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("Local TOC output truncated")
                content = choice["message"]["content"]
                # Persist only structured conclusions and concise evidence,
                # not private scratch reasoning.
                value = json.loads(content)
                reply = {"request_id": request["request_id"], "result": value,
                         "usage": result.get("usage", {})}
            except Exception as error:
                reply = {"request_id": request["request_id"], "error_type": type(error).__name__}
            print(json.dumps(reply, ensure_ascii=False), flush=True)
    finally:
        endpoint.close()


if __name__ == "__main__":
    main()
