import json
from collections.abc import Iterator
from typing import Any

import pytest
import requests
from conftest import STREAM_BODY, FakeOpenRouter, FakeResponse, delta, sse
from fastapi.testclient import TestClient

from free_router.adapters import anthropic
from free_router.adapters.anthropic import stream_to_anthropic
from free_router.routing import AllModelsFailed, route
from free_router.sse import SSEParser

MESSAGES = [{"role": "user", "content": "Hi"}]


def tool(index: int, **fields: Any) -> dict[str, Any]:
    return delta(tool_calls=[{"index": index, **fields}])


def finish(reason: str, **usage: int) -> dict[str, Any]:
    chunk: dict[str, Any] = {"choices": [{"delta": {}, "finish_reason": reason}]}
    if usage:
        chunk["usage"] = usage
    return chunk


def translate(body: bytes, split: int | None = None) -> list[dict[str, Any]]:
    """Run the translator, optionally feeding `split` bytes at a time."""

    size = split or len(body)
    chunks = [body[i : i + size] for i in range(0, len(body), size)]
    return list(stream_to_anthropic(chunks, "m"))


def types(events: list[dict[str, Any]]) -> list[str]:
    return [event["type"] for event in events]


# ---------------------------------------------------------------- SSE parser


def test_parser_handles_split_lines_crlf_comments_and_multiline() -> None:
    parser = SSEParser()
    body = b': hi\r\n\r\nevent: x\r\ndata: {"a":\r\ndata:1}\r\n\r\ndata: [DONE]\r\n\r\n'

    events = [event for byte in body for event in parser.feed(bytes([byte]))]

    assert events == ['{"a":\n1}', "[DONE]"]


def test_parser_flushes_an_unterminated_event() -> None:
    parser = SSEParser()

    assert parser.feed(b"data: last") == []
    assert parser.finish() == ["last"]


# ---------------------------------------------------------------- translator


def test_text_streams_as_it_arrives() -> None:
    events = translate(
        sse(
            delta(content="Hel"),
            delta(content="lo"),
            finish("stop", prompt_tokens=7, completion_tokens=2),
            "[DONE]",
        )
    )

    assert types(events) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert [e["delta"]["text"] for e in events[2:4]] == ["Hel", "lo"]
    assert events[5]["delta"]["stop_reason"] == "end_turn"
    assert events[5]["usage"] == {"input_tokens": 7, "output_tokens": 2}


def test_byte_by_byte_gives_the_same_events() -> None:
    body = sse(delta(content="Hel"), delta(content="lo"), finish("stop"), "[DONE]")

    whole, split = translate(body), translate(body, split=1)

    for events in (whole, split):
        for event in events:
            event.get("message", {}).pop("id", None)

    assert whole == split


def test_tool_calls_are_buffered_and_sent_whole() -> None:
    events = translate(
        sse(
            delta(content="Calling"),
            tool(0, id="c1", function={"name": "lookup", "arguments": '{"q"'}),
            tool(1, id="c2", function={"name": "broken", "arguments": "{oo"}),
            tool(0, function={"arguments": ": 1}"}),
            tool(1, function={"arguments": "ps"}),
            finish("tool_calls"),
            "[DONE]",
        )
    )

    blocks = [e["content_block"] for e in events if e["type"] == "content_block_start"]
    arguments = [
        json.loads(e["delta"]["partial_json"])
        for e in events
        if e["type"] == "content_block_delta"
        and e["delta"]["type"] == "input_json_delta"
    ]

    assert blocks == [
        {"type": "text", "text": ""},
        {"type": "tool_use", "id": "c1", "name": "lookup", "input": {}},
        {"type": "tool_use", "id": "c2", "name": "broken", "input": {}},
    ]
    assert arguments == [{"q": 1}, {}]
    assert [e["index"] for e in events if e["type"] == "content_block_stop"] == [
        0,
        1,
        2,
    ]
    assert events[-2]["delta"]["stop_reason"] == "tool_use"


def test_tool_calls_without_index_are_told_apart_by_id() -> None:
    events = translate(
        sse(
            delta(
                tool_calls=[{"id": "a", "function": {"name": "f", "arguments": "{"}}]
            ),
            delta(tool_calls=[{"function": {"arguments": "}"}}]),
            delta(tool_calls=[{"id": "b", "function": {"name": "g"}}]),
            finish("stop"),
        )
    )

    starts = [e["content_block"] for e in events if e["type"] == "content_block_start"]

    assert [(b["id"], b["name"]) for b in starts] == [("a", "f"), ("b", "g")]
    assert events[-2]["delta"]["stop_reason"] == "tool_use"


def test_empty_answer_still_has_a_block() -> None:
    events = translate(sse(finish("stop"), "[DONE]"))

    assert types(events) == [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


def test_upstream_error_mid_stream_becomes_an_error_event() -> None:
    events = translate(
        sse(
            delta(content="partial"),
            {"error": {"message": "provider died"}},
            delta(content="never sent"),
        )
    )

    assert events[-1] == anthropic.stream_error_event("Upstream error: provider died")
    assert "message_stop" not in types(events)


def test_pings_during_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(anthropic, "PING_INTERVAL", 0)

    events = list(
        stream_to_anthropic(
            [
                sse(delta(reasoning="thinking...")),
                b": keep-alive\n\n",
                sse(delta(content="hi"), finish("stop")),
            ],
            "m",
        )
    )

    assert types(events)[:3] == ["message_start", "ping", "ping"]
    assert "content_block_delta" in types(events)


def test_stream_without_done_still_finishes() -> None:
    events = translate(sse(delta(content="hi"), finish("length")))

    assert events[-2]["delta"]["stop_reason"] == "max_tokens"
    assert events[-1] == {"type": "message_stop"}


# ---------------------------------------------------------------- routing


@pytest.mark.parametrize(
    "bad_stream",
    [
        sse({"error": {"message": "overloaded", "code": 502}}),
        sse(delta(role="assistant", content=""), "[DONE]"),
        b"",
    ],
)
def test_streams_that_fail_before_output_fall_back(
    openrouter: FakeOpenRouter, bad_stream: bytes
) -> None:
    bad = FakeResponse(200, stream=bad_stream)
    openrouter.reply(bad, FakeResponse(200))

    result = route({"messages": MESSAGES}, stream=True)

    assert len(openrouter.calls) == 2
    assert result.model == openrouter.calls[1]
    assert bad.closed
    assert result.chunks is not None
    assert b"".join(result.chunks) == STREAM_BODY


def test_in_band_error_code_is_reported(openrouter: FakeOpenRouter) -> None:
    openrouter.reply(
        FakeResponse(200, stream=sse({"error": {"message": "slow down", "code": 429}}))
    )

    with pytest.raises(AllModelsFailed) as error:
        route({"messages": MESSAGES}, stream=True)

    assert error.value.attempts[0].status == 429
    assert error.value.attempts[0].error == "slow down"


# ---------------------------------------------------------------- endpoint


def anthropic_events(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def test_messages_stream_end_to_end(
    client: TestClient, openrouter: FakeOpenRouter
) -> None:
    openrouter.reply(
        FakeResponse(429, {}),
        FakeResponse(
            200,
            stream=sse(
                delta(content="Let me check."),
                tool(0, id="c1", function={"name": "f", "arguments": '{"a": 1}'}),
                finish("tool_calls", prompt_tokens=9, completion_tokens=4),
                "[DONE]",
            ),
        ),
    )

    response = client.post(
        "/v1/messages",
        json={
            "messages": MESSAGES,
            "stream": True,
            "tools": [{"name": "f", "input_schema": {"type": "object"}}],
        },
    )

    events = anthropic_events(response.text)

    assert response.status_code == 200
    assert response.headers["x-router-model"] == openrouter.calls[1]
    assert openrouter.payloads[1]["stream"] is True
    assert events[0]["type"] == "message_start"
    assert events[2]["delta"]["text"] == "Let me check."
    assert events[-2]["delta"]["stop_reason"] == "tool_use"
    assert events[-2]["usage"] == {"input_tokens": 9, "output_tokens": 4}


class BreaksOff(FakeResponse):
    def iter_content(self, chunk_size: int | None = None) -> Iterator[bytes]:
        yield sse(delta(content="partial"))
        raise requests.ConnectionError("reset")


def test_messages_stream_broken_mid_way_ends_with_error_event(
    client: TestClient, openrouter: FakeOpenRouter
) -> None:
    broken = BreaksOff(200)
    openrouter.reply(broken)

    response = client.post("/v1/messages", json={"messages": MESSAGES, "stream": True})

    events = anthropic_events(response.text)

    assert response.status_code == 200
    assert events[-1]["type"] == "error"
    assert any(e.get("delta", {}).get("text") == "partial" for e in events)
    assert broken.closed
