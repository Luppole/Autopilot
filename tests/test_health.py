import random

import pytest
from conftest import FakeOpenRouter, FakeResponse, completion
from fastapi.testclient import TestClient

from free_router import health
from free_router.routing import route

MESSAGES = [{"role": "user", "content": "Hi"}]


@pytest.mark.parametrize(
    ("status", "expected"),
    [(None, True), (200, True), (429, True), (408, True), (503, True), (400, False)],
)
def test_is_model_failure(status: int | None, expected: bool) -> None:
    assert health.is_model_failure(status) is expected


def test_cooldown_doubles_and_resets_on_success() -> None:
    health.record_failure("a")
    first = health.snapshot()["a"]["cooldown_seconds"]
    health.record_failure("a")
    second = health.snapshot()["a"]["cooldown_seconds"]

    assert first == pytest.approx(health.BASE_COOLDOWN, abs=1)
    assert second == pytest.approx(2 * health.BASE_COOLDOWN, abs=1)

    health.record_success("a", 1.5)

    stats = health.snapshot()["a"]
    assert stats["cooldown_seconds"] == 0
    assert (stats["successes"], stats["failures"]) == (1, 2)
    assert stats["latency_seconds"] == 1.5


def test_cooling_models_go_last_soonest_first() -> None:
    health.record_failure("slow")
    health.record_failure("slow")
    health.record_failure("fast")

    assert health.order(["slow", "fast", "ok"]) == ["ok", "fast", "slow"]


def test_reliable_models_tend_to_go_first() -> None:
    random.seed(0)
    for _ in range(10):
        health.record_success("good", 1)

    firsts = [health.order(["good", "new"])[0] for _ in range(200)]

    # Score ~0.95 vs 0.5 puts "good" first ~65% of the time.
    assert firsts.count("good") > 110


def test_failed_model_sits_out_the_next_request(openrouter: FakeOpenRouter) -> None:
    openrouter.reply(FakeResponse(429, {}), FakeResponse(200, completion()))
    route({"messages": MESSAGES})
    failed = openrouter.calls[0]

    openrouter.calls.clear()
    route({"messages": MESSAGES})

    assert openrouter.calls[0] != failed


def test_bad_request_does_not_cool_a_model_down(openrouter: FakeOpenRouter) -> None:
    openrouter.reply(FakeResponse(400, {}), FakeResponse(200, completion()))
    route({"messages": MESSAGES})

    assert health.snapshot().get(openrouter.calls[0], {}).get("failures", 0) == 0


def test_cooling_models_are_still_a_last_resort(openrouter: FakeOpenRouter) -> None:
    for model_id in ("acme/alpha:free", "acme/beta:free", "acme/gamma:free"):
        health.record_failure(model_id)

    result = route({"messages": MESSAGES})

    assert result.model == openrouter.calls[0]


def test_model_health_endpoint(client: TestClient, openrouter: FakeOpenRouter) -> None:
    route({"messages": MESSAGES})

    stats = client.get("/api/model-health").json()

    assert stats[openrouter.calls[0]]["successes"] == 1
