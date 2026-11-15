"""Transport regressions that do not load inference weights."""
import json
import asyncio

import httpx
import pytest

from src.model_api import OpenAIEndpoint
from src.model_router import ModelRouter
from src.toc_structure import LocalTocModel


def reply(content, finish_reason="stop"):
    return httpx.Response(200, json={"choices": [{"finish_reason": finish_reason,
        "message": {"content": json.dumps(content, ensure_ascii=False)}}], "usage": {"total_tokens": 5}})


@pytest.mark.parametrize("mode", ["json_schema", "llama_json", "json_object"])
def test_supported_structured_formats(monkeypatch, mode):
    sent = []
    def post(url, **kwargs):
        sent.append((url, kwargs))
        return reply({"value": 1})
    monkeypatch.setattr(httpx, "post", post)
    endpoint = OpenAIEndpoint("http://127.0.0.1:8081/v1/", "test-model", "secret-test-key", mode)
    schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
    endpoint.complete([{"role": "user", "content": "JSON"}], schema=schema)
    assert sent[0][0] == "http://127.0.0.1:8081/v1/chat/completions"
    format = sent[0][1]["json"]["response_format"]
    assert format["type"] == ("json_schema" if mode == "json_schema" else "json_object")
    if mode == "json_schema":
        assert format["json_schema"]["strict"] and format["json_schema"]["schema"] == schema
    if mode == "llama_json":
        assert format["schema"] == schema
    assert "secret-test-key" not in repr(endpoint) and "secret-test-key" not in json.dumps(endpoint.identity)


def test_api_only_local_router_grounds_and_reuses_cache(monkeypatch, tmp_path):
    calls = []
    def post(url, **kwargs):
        calls.append(url)
        return reply({"entities": [{"type": "PERIOD", "text": "宋代"}]})
    monkeypatch.setattr(httpx, "post", post)
    router = ModelRouter.from_settings(tmp_path, {"LOCAL_LLM_BASE_URL": "http://127.0.0.1:8081/v1",
        "LOCAL_LLM_MODEL": "deployed-model", "ALLOW_CLOUD_FALLBACK": "false"})
    assert router.local_available and router.local_model is None
    first = router.extract_entities("出土宋代瓷器")
    second = router.extract_entities("出土宋代瓷器")
    assert first["model"] == "deployed-model" and first["result"]["entities"][0]["start"] == 2
    assert second["cached"] and len(calls) == 1
    assert router.llm is None


def test_toc_adapter_uses_http_without_local_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: reply({"nodes": []}))
    model = LocalTocModel(None, tmp_path, settings={"LOCAL_LLM_MODEL": "deployed-model"})
    assert model.infer("request-1", "JSON", {"type": "object"})["result"] == {"nodes": []}
    model.close()


def test_timeout_truncation_and_invalid_response(monkeypatch):
    endpoint = OpenAIEndpoint("http://127.0.0.1:8081/v1", "test-model")
    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("test")
    monkeypatch.setattr(httpx, "post", timeout)
    with pytest.raises(TimeoutError):
        endpoint.complete([])
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: reply({}, "length"))
    with pytest.raises(ValueError, match="truncated"):
        endpoint.complete([])
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: httpx.Response(200, json={"choices": []}))
    with pytest.raises(ValueError, match="Malformed"):
        endpoint.complete([])


def test_local_server_adapts_standard_schema_and_tool_history():
    from src.model_server import OpenAICompatibility
    observed = []
    async def application(scope, receive, send):
        body = (await receive())["body"]
        observed.append((scope, json.loads(body)))
    async def send(event):
        observed.append(event)
    payload = {"model": "local", "messages": [{"role": "assistant", "content": None,
        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "test", "strict": True,
            "schema": {"type": "object", "properties": {"name": {"type": "string"}}}}}}
    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}
    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}
    asyncio.run(OpenAICompatibility(application, "local")(scope, receive, send))
    forwarded = observed[0][1]
    assert forwarded["messages"][0]["content"] == ""
    assert forwarded["response_format"] == {"type": "json_object", "schema": payload["response_format"]["json_schema"]["schema"]}
    observed.clear()
    payload["model"] = "not-served"
    asyncio.run(OpenAICompatibility(application, "local")(scope, receive, send))
    assert observed[0]["status"] == 404
    for messages in (None, "invalid", [], [None]):
        observed.clear()
        payload.update(model="local", messages=messages)
        asyncio.run(OpenAICompatibility(application, "local")(scope, receive, send))
        assert observed[0]["status"] == 400
