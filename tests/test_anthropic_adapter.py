import json
from typing import Any

import pytest
from conftest import completion

from free_router.adapters import anthropic
from free_router.errors import InvalidRequestError


def test_tool_round_trip_request() -> None:
    payload = anthropic.request_to_openai(
        {
            "system": [{"type": "text", "text": "Be brief."}],
            "max_tokens": 100,
            "stop_sequences": ["END"],
            "tools": [
                {"name": "lookup", "input_schema": {"type": "object"}},
                {"type": "web_search_20250305", "name": "web_search"},
            ],
            "tool_choice": {"type": "any"},
            "messages": [
                {"role": "user", "content": "Find it"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Looking"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "lookup",
                            "input": {"q": 1},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "found",
                        },
                        {"type": "text", "text": "Thanks"},
                    ],
                },
            ],
        }
    )

    assert payload["model"] == "auto"
    assert payload["max_tokens"] == 100
    assert payload["stop"] == ["END"]
    assert payload["tool_choice"] == "required"
    assert [t["function"]["name"] for t in payload["tools"]] == ["lookup"]

    roles = [message["role"] for message in payload["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "user"]

    call = payload["messages"][2]["tool_calls"][0]
    assert call["id"] == "t1"
    assert json.loads(call["function"]["arguments"]) == {"q": 1}
    assert payload["messages"][3]["tool_call_id"] == "t1"


def test_tool_only_assistant_turn_has_null_content() -> None:
    payload = anthropic.request_to_openai(
        {
            "messages": [
                {"role": "user", "content": "Go"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "f", "input": {}}
                    ],
                },
            ],
        }
    )

    assert payload["messages"][1]["content"] is None


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"messages": []},
        {"messages": "Hi"},
        {"messages": [{"role": "system", "content": "x"}]},
    ],
)
def test_rejects_malformed_requests(body: dict[str, Any]) -> None:
    with pytest.raises(InvalidRequestError):
        anthropic.request_to_openai(body)


def test_tolerates_odd_blocks() -> None:
    payload = anthropic.request_to_openai(
        {
            "tool_choice": "auto",
            "messages": [
                {"role": "user", "content": ["junk", {"type": "unknown"}, None]},
                {"role": "user", "content": None},
            ],
        }
    )

    assert [m["content"] for m in payload["messages"]] == ["", ""]
    assert "tool_choice" not in payload


def test_response_with_tool_call() -> None:
    message = anthropic.response_to_anthropic(
        completion(
            content="Calling",
            finish_reason="tool_calls",
            tool_calls=[
                {"id": "c1", "function": {"name": "lookup", "arguments": '{"q": 1}'}},
                {"id": "c2", "function": {"name": "broken", "arguments": "{oops"}},
                {"id": "c3"},
            ],
        ),
        "acme/alpha:free",
    )

    assert message["stop_reason"] == "tool_use"
    assert message["content"][1:] == [
        {"type": "tool_use", "id": "c1", "name": "lookup", "input": {"q": 1}},
        {"type": "tool_use", "id": "c2", "name": "broken", "input": {}},
    ]
    assert message["usage"] == {"input_tokens": 3, "output_tokens": 5}


def test_empty_reply_still_has_a_block() -> None:
    data = completion(content=None)
    data["usage"] = None

    message = anthropic.response_to_anthropic(data, "m")

    assert message["content"] == [{"type": "text", "text": ""}]
    assert message["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_list_content_is_joined() -> None:
    data = completion(
        content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    )

    assert anthropic.response_to_anthropic(data, "m")["content"][0]["text"] == "ab"


def test_tool_use_stop_reason_even_if_provider_says_stop() -> None:
    message = anthropic.response_to_anthropic(
        completion(
            tool_calls=[{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
            finish_reason="stop",
        ),
        "m",
    )

    assert message["stop_reason"] == "tool_use"
