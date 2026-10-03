from typing import Any

import pytest
from conftest import FakeOpenRouter

from free_router.capabilities import estimate_prompt_tokens, fit_payload, unfit_reason
from free_router.config import settings
from free_router.errors import InvalidRequestError
from free_router.routing import route
from free_router.storage import write_json

MESSAGES = [{"role": "user", "content": "Hi"}]
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
IMAGE = [
    {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "data:,"}}],
    }
]


def text_model(**fields: Any) -> dict[str, Any]:
    return {
        "id": "a/text:free",
        "supported_parameters": ["max_tokens"],
        "architecture": {"input_modalities": ["text"]},
        "context_length": 1000,
        "top_provider": {"max_completion_tokens": 500},
        **fields,
    }


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"messages": MESSAGES}, None),
        ({"messages": MESSAGES, "tools": TOOLS}, "doesn't support tools"),
        ({"messages": IMAGE}, "doesn't accept images"),
        ({"messages": [{"role": "user", "content": "x" * 8000}]}, "context"),
    ],
)
def test_unfit_reason(payload: dict[str, Any], reason: str | None) -> None:
    result = unfit_reason(text_model(), payload, estimate_prompt_tokens(payload))

    if reason is None:
        assert result is None
    else:
        assert result is not None and reason in result


def test_models_without_metadata_are_assumed_capable() -> None:
    payload = {"messages": IMAGE, "tools": TOOLS}

    assert unfit_reason({"id": "old:free"}, payload, 10**9) is None


def test_fit_payload_lowers_output_limit() -> None:
    payload = {"messages": MESSAGES, "max_tokens": 32000, "max_completion_tokens": 9}

    fitted = fit_payload(text_model(), payload, prompt_tokens=100)

    assert fitted["max_tokens"] == 500
    assert fitted["max_completion_tokens"] == 9
    assert payload["max_tokens"] == 32000


def test_fit_payload_leaves_room_for_the_prompt() -> None:
    fitted = fit_payload(text_model(), {"max_tokens": 32000}, prompt_tokens=800)

    assert fitted["max_tokens"] == 200


def test_routes_tool_requests_to_tool_models(openrouter: FakeOpenRouter) -> None:
    write_json(
        settings.models_file,
        [
            text_model(id="a/plain:free"),
            text_model(id="a/tools:free", supported_parameters=["tools"]),
        ],
    )

    for _ in range(5):
        route({"messages": MESSAGES, "tools": TOOLS, "max_tokens": 32000})

    assert set(openrouter.calls) == {"a/tools:free"}
    assert openrouter.payloads[0]["max_tokens"] == 500


def test_refuses_when_no_model_fits(openrouter: FakeOpenRouter) -> None:
    write_json(settings.models_file, [text_model()])

    with pytest.raises(InvalidRequestError, match="doesn't support tools"):
        route({"messages": MESSAGES, "tools": TOOLS})

    assert openrouter.calls == []


def test_explicit_model_skips_capability_checks(openrouter: FakeOpenRouter) -> None:
    write_json(settings.models_file, [text_model()])

    route({"model": "a/text:free", "messages": MESSAGES, "tools": TOOLS})

    assert openrouter.calls == ["a/text:free"]
