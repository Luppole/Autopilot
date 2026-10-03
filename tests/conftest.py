import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from free_router import health, routing
from free_router.api.app import app
from free_router.config import settings
from free_router.storage import write_json

MODELS = [
    {"id": "acme/alpha:free", "name": "Acme: Alpha (free)"},
    {"id": "acme/beta:free", "name": "Acme: Beta (free)"},
    {"id": "acme/gamma:free", "name": "Acme: Gamma (free)"},
]

STREAM_BODY = b'event: x\ndata: {"delta":"hi"}\n\n: keep-alive\n\ndata: [DONE]\n\n'


class FakeResponse:
    def __init__(self, status_code: int, body: Any = None, raw: bytes | None = None):
        self.status_code = status_code
        self.ok = status_code < 400
        self._body = body
        self.text = raw.decode() if raw is not None else json.dumps(body)
        self.closed = False
        # Set by FakeOpenRouter: a streaming request gets STREAM_BODY.
        self.streaming = False

    def json(self) -> Any:
        return json.loads(self.text)

    def close(self) -> None:
        self.closed = True

    def iter_content(self, chunk_size: int | None = None) -> Iterator[bytes]:
        body = STREAM_BODY if self.streaming else self.text.encode()
        yield body[:10]
        yield body[10:]


def completion(
    content: Any = "hi",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}

    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5},
    }


class FakeOpenRouter:
    """A queue of canned responses; the last one repeats."""

    def __init__(self) -> None:
        self.responses = [FakeResponse(200, completion())]
        self.calls: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self.timeouts: list[float] = []

    def reply(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)

    def __call__(self, payload: dict[str, Any], timeout: float) -> FakeResponse:
        self.calls.append(payload["model"])
        self.payloads.append(payload)
        self.timeouts.append(timeout)

        response = (
            self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        )
        response.streaming = bool(payload.get("stream"))
        return response


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own data folder, seeded with MODELS."""

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "rate_limit_delay", 0)
    monkeypatch.setattr(settings, "max_attempts", 0)
    monkeypatch.setattr(settings, "request_deadline", 0)
    write_json(settings.models_file, MODELS)
    health.reset()

    return tmp_path


@pytest.fixture
def openrouter(monkeypatch: pytest.MonkeyPatch) -> FakeOpenRouter:
    fake = FakeOpenRouter()
    monkeypatch.setattr(routing, "post_chat", fake)
    return fake


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")
