"""OpenAI-compatible HTTP transport for independently deployed inference services."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx


@dataclass
class OpenAIEndpoint:
    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    response_format: str = "json_schema"
    timeout: float = 150
    attempts: int = 1

    def __post_init__(self):
        self.base_url = self.base_url.rstrip("/")
        parsed = urlsplit(self.base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment):
            raise ValueError("Model base_url must be an HTTP(S) endpoint without credentials or query parameters")
        if not self.model or self.response_format not in {"json_schema", "llama_json", "json_object"}:
            raise ValueError("Invalid model or structured-output mode")
        if self.timeout <= 0 or not 1 <= self.attempts <= 3:
            raise ValueError("Invalid model timeout or retry count")

    @property
    def identity(self):
        return {"endpoint": self.base_url, "model": self.model, "format": self.response_format}

    def complete(self, messages, *, schema=None, max_tokens=1200, before_request=None,
                 on_response=None, extra=None):
        payload = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        if schema is not None:
            if self.response_format == "json_schema":
                payload["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "extraction", "schema": schema, "strict": True,
                }}
            elif self.response_format == "llama_json":
                payload["response_format"] = {"type": "json_object", "schema": schema}
            else:
                payload["response_format"] = {"type": "json_object"}
        payload.update(extra or {})
        headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else {}
        for attempt in range(self.attempts):
            if before_request:
                before_request()
            try:
                response = httpx.post(self.base_url + "/chat/completions", headers=headers,
                                      json=payload, timeout=self.timeout,
                                      trust_env=urlsplit(self.base_url).hostname not in {"localhost", "127.0.0.1", "::1"})
                if response.status_code in {429, 500, 502, 503, 504} and attempt + 1 < self.attempts:
                    time.sleep(2 ** attempt)
                    continue
                if response.status_code != 200:
                    raise RuntimeError(f"Model endpoint returned HTTP {response.status_code}")
                value = response.json()
                if on_response:
                    on_response(value)
                choice = value["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("Model output truncated")
                if choice["message"].get("refusal"):
                    raise ValueError("Model declined this request")
                if not isinstance(choice["message"].get("content"), str):
                    raise ValueError("Model response has no text content")
                return value
            except httpx.TimeoutException:
                if attempt + 1 == self.attempts:
                    raise TimeoutError("Model endpoint exceeded its time limit") from None
            except httpx.NetworkError:
                if attempt + 1 == self.attempts:
                    raise RuntimeError("Model endpoint unavailable") from None
            except (KeyError, IndexError, json.JSONDecodeError):
                raise ValueError("Malformed model response") from None
            time.sleep(2 ** attempt)
        raise RuntimeError("Model retries exhausted")

    def infer(self, request_id, prompt, schema, *, max_tokens=1400):
        value = self.complete([{"role": "user", "content": prompt}], schema=schema, max_tokens=max_tokens)
        return {"request_id": request_id, "result": json.loads(value["choices"][0]["message"]["content"]),
                "usage": value.get("usage", {})}

    def close(self):
        """The HTTP client does not own the external inference process."""
