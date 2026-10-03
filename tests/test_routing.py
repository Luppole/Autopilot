from typing import Any

import pytest
import requests
from conftest import FakeOpenRouter, FakeResponse, completion

from free_router.config import settings
from free_router.errors import ConfigError, InvalidRequestError, NoModelsError
from free_router.routing import AllModelsFailed, route
from free_router.storage import write_json

MESSAGES = [{"role": "user", "content": "Hi"}]


def test_skips_failing_models(openrouter: FakeOpenRouter) -> None:
    busy = FakeResponse(429, {"error": "busy"})
    openrouter.reply(
        busy,
        FakeResponse(503, {"error": "down"}),
        FakeResponse(200, completion("answer")),
    )

    result = route({"messages": MESSAGES})

    assert len(openrouter.calls) == 3
    assert len(set(openrouter.calls)) == 3
    assert result.model == openrouter.calls[-1]
    assert result.data is not None and result.data["model"] == result.model
    assert [a.status for a in result.attempts] == [429, 503]
    assert busy.closed


@pytest.mark.parametrize(
    "bad_reply",
    [
        FakeResponse(200, {"error": "provider hiccup"}),
        FakeResponse(200, {"choices": []}),
        FakeResponse(200, raw=b"<html>gateway</html>"),
    ],
)
def test_skips_unusable_success_replies(
    openrouter: FakeOpenRouter, bad_reply: FakeResponse
) -> None:
    openrouter.reply(bad_reply, FakeResponse(200, completion()))

    result = route({"messages": MESSAGES})

    assert len(result.attempts) == 1


def test_network_errors_are_skipped(
    openrouter: FakeOpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def flaky(payload: dict[str, Any], timeout: float) -> FakeResponse:
        calls.append(payload["model"])
        if len(calls) == 1:
            raise requests.ConnectionError("reset")
        return FakeResponse(200, completion())

    monkeypatch.setattr("free_router.routing.post_chat", flaky)

    result = route({"messages": MESSAGES})

    assert result.attempts[0].status is None
    assert result.attempts[0].error == "reset"


def test_explicit_free_model_is_tried_once(openrouter: FakeOpenRouter) -> None:
    openrouter.reply(FakeResponse(429, {"error": "busy"}))

    with pytest.raises(AllModelsFailed):
        route({"model": "acme/beta:free", "messages": MESSAGES})

    assert openrouter.calls == ["acme/beta:free"]


def test_paid_or_unknown_model_is_refused(openrouter: FakeOpenRouter) -> None:
    with pytest.raises(InvalidRequestError):
        route({"model": "openai/gpt-5", "messages": MESSAGES})

    assert openrouter.calls == []


@pytest.mark.parametrize("messages", [None, [], "Hi"])
def test_requires_messages(openrouter: FakeOpenRouter, messages: object) -> None:
    with pytest.raises(InvalidRequestError):
        route({"messages": messages})

    assert openrouter.calls == []


def test_reports_every_failure(openrouter: FakeOpenRouter) -> None:
    openrouter.reply(FakeResponse(429, {"error": "x" * 5000}))

    with pytest.raises(AllModelsFailed) as error:
        route({"messages": MESSAGES})

    assert len(error.value.attempts) == 3
    assert len(error.value.attempts[0].error) < 600
    assert error.value.to_dict()["type"] == "all_models_failed"


def test_max_attempts(
    openrouter: FakeOpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "max_attempts", 2)
    openrouter.reply(FakeResponse(429, {}))

    with pytest.raises(AllModelsFailed):
        route({"messages": MESSAGES})

    assert len(openrouter.calls) == 2


def test_no_models_synced(openrouter: FakeOpenRouter) -> None:
    write_json(settings.models_file, [])

    with pytest.raises(NoModelsError):
        route({"messages": MESSAGES})


def test_missing_api_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ConfigError):
        route({"messages": MESSAGES})
