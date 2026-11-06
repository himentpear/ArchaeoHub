"""Persistent local-only llama.cpp worker with bounded JSON conclusions."""
import argparse
import json
import sys


def main():
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1200)
    args = parser.parse_args()
    from llama_cpp import Llama
    llm = Llama(model_path=args.model, n_ctx=args.context, n_threads=args.threads,
                n_threads_batch=args.threads, n_batch=256, verbose=False)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            try:
                result = llm.create_chat_completion(
                    messages=[{"role": "user", "content": request["prompt"]}],
                    response_format={"type": "json_object", "schema": request["schema"]},
                    max_tokens=args.max_tokens, temperature=0, seed=42,
                )
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
        llm.close()


if __name__ == "__main__":
    main()
