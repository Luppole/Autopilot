import json

import pytest
from conftest import STREAM_BODY, FakeOpenRouter, FakeResponse, completion
from fastapi.testclient import TestClient

from free_router.api.app import app
from free_router.errors import UpstreamError

CHAT = {"messages": [{"role": "user", "content": "Hi"}]}


def test_chat_completion_reports_routing(
    client: TestClient, openrouter: FakeOpenRouter
) -> None:
    openrouter.reply(
        FakeResponse(429, {"error": "busy"}), FakeResponse(200, completion())
    )

    response = client.post("/v1/chat/completions", json=CHAT)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hi"
    assert response.headers["x-router-model"] == openrouter.calls[-1]
    assert json.loads(response.headers["x-router-attempts"]) == [
        {"model": openrouter.calls[0], "status": 429},
    ]


def test_stream_is_relayed_byte_for_byte(
    client: TestClient, openrouter: FakeOpenRouter
) -> None:
    response = client.post("/v1/chat/completions", json={**CHAT, "stream": True})

    assert response.status_code == 200
    assert response.content == STREAM_BODY


def test_all_models_failing_uses_openai_error_shape(
    client: TestClient, openrouter: FakeOpenRouter
) -> None:
    openrouter.reply(FakeResponse(429, {"error": "busy"}))

    response = client.post("/v1/chat/completions", json=CHAT)

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["type"] == "all_models_failed"
    assert len(error["attempts"]) == 3


def test_bad_requests_get_400(client: TestClient, openrouter: FakeOpenRouter) -> None:
    assert client.post("/v1/chat/completions", json={}).status_code == 400
    assert client.post("/v1/chat/completions", json=[1, 2]).status_code == 400
    assert (
        client.post(
            "/v1/chat/completions", json={**CHAT, "model": "paid/x"}
        ).status_code
        == 400
    )
    assert openrouter.calls == []


def test_anthropic_messages(client: TestClient, openrouter: FakeOpenRouter) -> None:
    response = client.post(
        "/v1/messages",
        json={"model": "claude-anything", "max_tokens": 10, **CHAT},
    )

    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "hi"}]


def test_anthropic_messages_stream(
    client: TestClient, openrouter: FakeOpenRouter
) -> None:
    response = client.post("/v1/messages", json={**CHAT, "stream": True})

    assert response.status_code == 200
    assert response.text.startswith("event: message_start\n")
    assert response.text.rstrip().endswith('data: {"type":"message_stop"}')


def test_anthropic_errors_use_anthropic_shape(
    client: TestClient, openrouter: FakeOpenRouter
) -> None:
    openrouter.reply(FakeResponse(429, {}))

    failed = client.post("/v1/messages", json=CHAT)
    invalid = client.post("/v1/messages", json={"messages": []})

    assert failed.status_code == 503
    assert failed.json()["type"] == "error"
    assert failed.json()["error"]["type"] == "overloaded_error"
    assert invalid.status_code == 400
    assert invalid.json()["error"]["type"] == "invalid_request_error"


def test_non_json_posts_are_refused(
    client: TestClient, openrouter: FakeOpenRouter
) -> None:
    response = client.post(
        "/v1/chat/completions",
        content=json.dumps(CHAT),
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 415
    assert openrouter.calls == []


def test_unknown_hosts_are_refused() -> None:
    rebound = TestClient(app, base_url="http://attacker.example")

    assert rebound.get("/health").status_code == 400


def test_sync_failure_is_a_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def offline() -> None:
        raise UpstreamError("Couldn't fetch the model list from OpenRouter: offline")

    monkeypatch.setattr("free_router.api.dashboard.sync_free_models", offline)

    response = client.post("/api/sync", json={})

    assert response.status_code == 502
    assert "offline" in response.json()["error"]["message"]


def test_unexpected_errors_are_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise ValueError("secret internals")

    monkeypatch.setattr("free_router.api.dashboard.load_history", boom)
    client = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)

    response = client.get("/api/history")

    assert response.status_code == 500
    assert "secret" not in response.text


def test_unknown_routes_use_error_shape(client: TestClient) -> None:
    response = client.get("/v1/nope")

    assert response.status_code == 404
    assert "message" in response.json()["error"]


def test_models_and_dashboard(client: TestClient) -> None:
    assert len(client.get("/v1/models").json()["data"]) == 3
    assert client.get("/health").json() == {"status": "ok", "free_models": 3}
    assert client.get("/api/status").json()["free_models"] == 3
    assert "Free Model Router" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
