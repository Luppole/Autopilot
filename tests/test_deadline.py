from collections.abc import Iterator
from typing import Any

import pytest
from conftest import FakeOpenRouter, FakeResponse, completion
from fastapi.testclient import TestClient

from free_router import health, openrouter, routing
from free_router.config import settings
from free_router.routing import DeadlineExceeded, route

MESSAGES = [{"role": "user", "content": "Hi"}]


class Clock:
    """Stands in for the `time` module inside routing."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(routing, "time", fake)
    monkeypatch.setattr(settings, "request_deadline", 100)
    return fake


def test_attempt_timeout_is_capped_by_time_left(
    openrouter: FakeOpenRouter, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "request_timeout", 300)
    openrouter.reply(FakeResponse(503, {}), FakeResponse(200, completion()))
    real_call = openrouter.__call__

    def slow(payload: dict[str, Any], timeout: float) -> FakeResponse:
        clock.now += 30
        return real_call(payload, timeout)

    monkeypatch.setattr(routing, "post_chat", slow)

    route({"messages": MESSAGES})

    assert openrouter.timeouts == [100, 70]


def test_gives_up_when_the_deadline_passes(
    clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def hangs(payload: dict[str, Any], timeout: float) -> FakeResponse:
        calls.append(payload["model"])
        clock.now += 60
        return FakeResponse(503, {})

    monkeypatch.setattr(routing, "post_chat", hangs)

    with pytest.raises(DeadlineExceeded) as error:
        route({"messages": MESSAGES})

    assert len(calls) == 2
    assert error.value.status_code == 504
    assert "within 100s" in error.value.message

    # The first model failed on its own; the second was cut off by us.
    stats = health.snapshot()
    assert stats[calls[0]]["failures"] == 1
    assert calls[1] not in stats


class TrickleResponse(FakeResponse):
    def __init__(self, clock: Clock) -> None:
        super().__init__(200, completion())
        self.clock = clock

    def iter_content(self, chunk_size: int | None = None) -> Iterator[bytes]:
        while True:
            self.clock.now += 30
            yield b" "


def test_slow_trickling_body_is_cut_off(
    openrouter: FakeOpenRouter, clock: Clock
) -> None:
    trickle = TrickleResponse(clock)
    openrouter.reply(trickle)

    with pytest.raises(DeadlineExceeded) as error:
        route({"messages": MESSAGES})

    assert error.value.attempts[0].error == "Request deadline reached mid-reply"
    assert trickle.closed


def test_no_deadline_when_disabled(
    openrouter: FakeOpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "request_deadline", 0)
    monkeypatch.setattr(settings, "request_timeout", 300)

    route({"messages": MESSAGES})

    assert openrouter.timeouts == [300]


@pytest.mark.parametrize(
    ("path", "error_type"),
    [("/v1/messages", "timeout_error"), ("/v1/chat/completions", None)],
)
def test_deadline_error_shape(
    client: TestClient,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    error_type: str | None,
) -> None:
    def hangs(payload: dict[str, Any], timeout: float) -> FakeResponse:
        clock.now += 200
        return FakeResponse(503, {})

    monkeypatch.setattr(routing, "post_chat", hangs)

    response = client.post(path, json={"messages": MESSAGES})

    assert response.status_code == 504
    error = response.json()["error"]
    assert error["type"] == (error_type or "deadline_exceeded")


def test_post_chat_uses_short_connect_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, Any] = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(settings, "connect_timeout", 10)
    monkeypatch.setattr(
        openrouter._session, "post", lambda url, **kwargs: sent.update(kwargs)
    )

    openrouter.post_chat({"model": "x"}, timeout=120)
    assert sent["timeout"] == (10, 120)
    assert sent["stream"] is True

    openrouter.post_chat({"model": "x"}, timeout=4)
    assert sent["timeout"] == (4, 4)
