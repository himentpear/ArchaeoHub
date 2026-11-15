"""Run local GGUF inference as an independent OpenAI-compatible service."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.settings import load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    settings = load_settings(args.settings)
    path = args.model or Path(settings.get("LOCAL_MODEL_PATH", "data/processed/models/Qwen3-4B-Q4_K_M.gguf"))
    if not path.is_file():
        parser.error("Local GGUF model file not found")
    import uvicorn
    from src.model_server import OpenAICompatibility
    from llama_cpp.server.app import create_app
    from llama_cpp.server.settings import ModelSettings, ServerSettings

    server = ServerSettings(host="127.0.0.1", port=args.port, api_key=settings.get("LOCAL_LLM_API_KEY") or None)
    model = ModelSettings(model=str(path.resolve()), model_alias=settings.get("LOCAL_LLM_MODEL") or path.name,
                          n_ctx=args.context, n_threads=args.threads, n_threads_batch=args.threads,
                          n_batch=256, verbose=False)
    app = create_app(server_settings=server, model_settings=[model])

    app.add_middleware(OpenAICompatibility, model_name=model.model_alias)
    uvicorn.run(app, host=server.host, port=server.port)


if __name__ == "__main__":
    main()
