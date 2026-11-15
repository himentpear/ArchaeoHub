"""Bounded PAI pilot jobs running in their own dependency environment."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import uuid

from src.ingest.page_pipeline import atomic_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = ROOT / "data/processed/models/agent-runtime" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


class AgentRuns:
    def __init__(self, directory, runtime=DEFAULT_RUNTIME):
        self.directory, self.runtime = Path(directory), Path(runtime)
        self._slot = threading.BoundedSemaphore(1)
        self._lock = threading.Lock()
        self._process = None
        self._active_id = None
        self._closed = False

    def start(self, query, library_url, namespace="", artifact_id=""):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 128:
            raise ValueError("输入1至128字的器号")
        if not isinstance(namespace, str) or len(namespace) > 128:
            raise ValueError("Invalid report namespace")
        if not isinstance(artifact_id, str) or (artifact_id and not re.fullmatch(r"[a-fA-F0-9-]{36}", artifact_id)):
            raise ValueError("Invalid artifact identifier")
        if not self.runtime.is_file():
            raise ValueError("Agent运行环境尚未安装，请按部署文档安装requirements-agent.txt")
        if not self._slot.acquire(blocking=False):
            if self._active_id:
                current = self.get(self._active_id)
                if (current["query"] == query.strip() and current["namespace"] == namespace
                        and current.get("artifact_id", "") == artifact_id):
                    return current
            raise ValueError("Agent正在核查另一件器物，请稍后再试")
        state = {"id": uuid.uuid4().hex, "query": query.strip(), "namespace": namespace, "artifact_id": artifact_id,
                 "status": "running", "started_at": time.time()}
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            atomic_json(self.directory / (state["id"] + ".json"), state)
            self._active_id = state["id"]
            threading.Thread(target=self._run, args=(state, library_url), daemon=True).start()
        except Exception:
            self._slot.release()
            raise
        return state

    def _run(self, state, library_url):
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Library service closed")
                process = subprocess.Popen([str(self.runtime), str(ROOT / "scripts/check_artifact_agent.py"),
                    "--query", state["query"], "--library-url", library_url, "--namespace", state["namespace"],
                    "--artifact-id", state["artifact_id"]],
                    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), start_new_session=os.name != "nt")
                self._process = process
            try:
                stdout, _ = process.communicate(timeout=240)
            except subprocess.TimeoutExpired:
                self._terminate(process)
                raise TimeoutError("Agent核查超过240秒时限") from None
            if not stdout.strip():
                raise RuntimeError("Agent worker exited without a result")
            result = json.loads(stdout.strip().splitlines()[-1])
            if result.get("status") not in {"completed", "failed", "no_matches", "stale"}:
                raise ValueError("Invalid agent result")
            state.update(result)
        except Exception as error:
            state.update(status="failed", error_type=type(error).__name__, error=str(error)[:500])
        finally:
            state["finished_at"] = time.time()
            try:
                atomic_json(self.directory / (state["id"] + ".json"), state)
            finally:
                with self._lock:
                    self._process = None
                    self._active_id = None
                self._slot.release()

    def get(self, identifier):
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("Invalid agent run identifier")
        path = self.directory / (identifier + ".json")
        if not path.is_file():
            raise LookupError("Agent run not found")
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["status"] == "running" and self._active_id != identifier:
            state.update(status="interrupted", error="阅览服务已重启，请重新发起核查。")
        return state

    @staticmethod
    def _terminate(process):
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)

    def close(self):
        with self._lock:
            self._closed = True
            if self._process:
                self._terminate(self._process)
