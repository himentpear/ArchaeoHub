"""Regression for bounded jobs and reconnecting to an ongoing evidence check."""
import json
import threading
import time

import pytest

from src.agent_service import AgentRuns


def test_agent_job_reconnects_without_starting_a_second_process(tmp_path, monkeypatch):
    finished = threading.Event()
    launches = []
    class Process:
        def __init__(self, command, **kwargs):
            launches.append(command)
        def communicate(self, timeout):
            finished.wait(3)
            return json.dumps({"status": "completed", "output": "source-pair test", "evidence": []}), ""
        def poll(self):
            return 0
    monkeypatch.setattr("src.agent_service.subprocess.Popen", Process)
    runtime = tmp_path / "python-placeholder"
    runtime.write_text("test-only", encoding="utf-8")
    jobs = AgentRuns(tmp_path / "runs", runtime=runtime)
    try:
        first = jobs.start("T1∶1", "http://127.0.0.1:1234", "test")
        second = jobs.start("T1∶1", "http://127.0.0.1:1234", "test")
        assert first["id"] == second["id"]
        with pytest.raises(ValueError, match="正在"):
            jobs.start("T1∶2", "http://127.0.0.1:1234", "test")
        with pytest.raises(ValueError, match="正在"):
            jobs.start("T1∶1", "http://127.0.0.1:1234", "test", "00000000-0000-0000-0000-000000000001")
        finished.set()
        deadline = time.monotonic() + 3
        while jobs.get(first["id"])["status"] == "running" and time.monotonic() < deadline:
            time.sleep(.01)
        assert jobs.get(first["id"])["status"] == "completed" and len(launches) == 1
        with pytest.raises(ValueError):
            jobs.get("../outside")
    finally:
        finished.set()
        jobs.close()
