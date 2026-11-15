"""Translate standard OpenAI requests at the local inference-service boundary."""
import json


class OpenAICompatibility:
    def __init__(self, app, model_name):
        self.app, self.model_name = app, model_name

    async def error(self, send, status, message):
        body = json.dumps({"error": {"message": message, "type": "invalid_request_error"}}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") != "/v1/chat/completions":
            return await self.app(scope, receive, send)
        chunks = []
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            chunks.append(event.get("body", b""))
            if not event.get("more_body", False):
                break
        try:
            payload = json.loads(b"".join(chunks))
        except (ValueError, UnicodeDecodeError):
            return await self.error(send, 400, "Invalid JSON request")
        if not isinstance(payload, dict):
            return await self.error(send, 400, "JSON object required")
        # The upstream proxy silently selects its default for unknown aliases.
        if payload.get("model") is not None and payload["model"] != self.model_name:
            return await self.error(send, 404, "Requested model is not served")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages or any(not isinstance(message, dict) for message in messages):
            return await self.error(send, 400, "messages must be a non-empty array of objects")
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls") and message.get("content") is None:
                message["content"] = ""
        format = payload.get("response_format")
        if isinstance(format, dict) and format.get("type") == "json_schema":
            specification = format.get("json_schema", {})
            if not isinstance(specification, dict) or not isinstance(specification.get("schema"), dict):
                return await self.error(send, 400, "JSON schema required")
            payload["response_format"] = {"type": "json_object", "schema": specification["schema"]}
        body = json.dumps(payload, ensure_ascii=False).encode()
        forwarded = False

        async def replay():
            nonlocal forwarded
            if not forwarded:
                forwarded = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        headers = [(key, value) for key, value in scope.get("headers", []) if key.lower() != b"content-length"]
        await self.app({**scope, "headers": headers + [(b"content-length", str(len(body)).encode())]}, replay, send)
