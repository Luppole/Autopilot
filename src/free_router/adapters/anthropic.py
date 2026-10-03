"""Anthropic Messages API <-> OpenAI chat completions.

Input is client-supplied JSON, so every helper tolerates missing or
oddly-typed fields instead of trusting the shape.
"""

import json
import uuid
from collections.abc import Iterable, Iterator
from typing import Any

from free_router.errors import InvalidRequestError

Json = dict[str, Any]


def _blocks(value: Any) -> list[Json]:
    if not isinstance(value, list):
        return []
    return [block for block in value if isinstance(block, dict)]


# ============================================================
# Request: Anthropic -> OpenAI
# ============================================================


def content_to_openai(content: Any) -> Any:
    """Convert Anthropic content blocks into OpenAI-style content."""

    if not isinstance(content, list):
        return "" if content is None else content

    result: list[Json] = []

    for block in _blocks(content):
        block_type = block.get("type")

        if block_type == "text":
            result.append({"type": "text", "text": str(block.get("text", ""))})

        elif block_type == "image":
            source = block.get("source")

            if isinstance(source, dict) and source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                result.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    }
                )

            elif isinstance(source, dict) and source.get("type") == "url":
                result.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": source.get("url", "")},
                    }
                )

    if not result:
        return ""

    if len(result) == 1 and result[0]["type"] == "text":
        return result[0]["text"]

    return result


def tool_use_to_openai(block: Json) -> Json:
    return {
        "id": str(block.get("id") or f"call_{uuid.uuid4().hex}"),
        "type": "function",
        "function": {
            "name": str(block.get("name", "")),
            "arguments": json.dumps(block.get("input") or {}),
        },
    }


def system_to_text(system: Any) -> str:
    if isinstance(system, str):
        return system

    return "\n".join(
        str(block.get("text", ""))
        for block in _blocks(system)
        if block.get("type") == "text"
    )


def messages_to_openai(body: Json) -> list[Json]:
    messages: list[Json] = []

    system = system_to_text(body.get("system"))

    if system:
        messages.append({"role": "system", "content": system})

    for message in _blocks(body.get("messages")):
        role = message.get("role")

        if role not in ("user", "assistant"):
            raise InvalidRequestError(
                f'Message role must be "user" or "assistant", got {role!r}.'
            )

        content = message.get("content")

        if not isinstance(content, list):
            messages.append({"role": role, "content": content_to_openai(content)})
            continue

        blocks = _blocks(content)
        tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
        tool_results = [b for b in blocks if b.get("type") == "tool_result"]
        other = [b for b in blocks if b.get("type") not in ("tool_use", "tool_result")]

        # Tool results must directly follow the assistant's tool calls.
        for block in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id", "")),
                    "content": content_to_openai(block.get("content")),
                }
            )

        if other or tool_uses or not tool_results:
            converted: Json = {"role": role, "content": content_to_openai(other)}

            if tool_uses:
                converted["tool_calls"] = [tool_use_to_openai(b) for b in tool_uses]
                converted["content"] = converted["content"] or None

            messages.append(converted)

    return messages


def tools_to_openai(tools: Any) -> list[Json]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {"type": "object"},
            },
        }
        for tool in _blocks(tools)
        # Server tools (web search, etc.) have their own types and can't be proxied.
        if tool.get("type") in (None, "custom") and tool.get("name")
    ]


def tool_choice_to_openai(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return None

    choice_type = tool_choice.get("type")

    if choice_type == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name")}}

    return {"auto": "auto", "any": "required", "none": "none"}.get(str(choice_type))


def request_to_openai(body: Json) -> Json:
    if not _blocks(body.get("messages")):
        raise InvalidRequestError("`messages` must be a non-empty list.")

    payload: Json = {
        # The client's Claude model name means nothing upstream.
        "model": "auto",
        "messages": messages_to_openai(body),
    }

    for key in ("temperature", "top_p", "max_tokens"):
        if key in body:
            payload[key] = body[key]

    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]

    tools = tools_to_openai(body.get("tools"))

    if tools:
        payload["tools"] = tools
        tool_choice = tool_choice_to_openai(body.get("tool_choice"))

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    return payload


# ============================================================
# Response: OpenAI -> Anthropic
# ============================================================

STOP_REASONS = {
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "stop": "end_turn",
}


def _parse_arguments(arguments: Any) -> Json:
    if isinstance(arguments, dict):
        return arguments

    try:
        parsed = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    # Some providers return content as a list of parts.
    return "".join(
        str(part.get("text", ""))
        for part in _blocks(content)
        if part.get("type") == "text"
    )


def response_to_anthropic(data: Json, model: str) -> Json:
    choice = data["choices"][0]
    message = choice.get("message")

    if not isinstance(message, dict):
        message = {}

    content: list[Json] = []
    text = _message_text(message.get("content"))

    if text:
        content.append({"type": "text", "text": text})

    for tool_call in _blocks(message.get("tool_calls")):
        function = tool_call.get("function")

        if not isinstance(function, dict):
            continue

        content.append(
            {
                "type": "tool_use",
                "id": str(tool_call.get("id") or f"toolu_{uuid.uuid4().hex}"),
                "name": str(function.get("name", "")),
                "input": _parse_arguments(function.get("arguments")),
            }
        )

    # Anthropic clients expect at least one block.
    if not content:
        content.append({"type": "text", "text": ""})

    usage = data.get("usage")

    if not isinstance(usage, dict):
        usage = {}

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": STOP_REASONS.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


# ============================================================
# Streaming: replay a finished message as Anthropic SSE events
# ============================================================


def stream_events(message: Json) -> Iterator[Json]:
    yield {
        "type": "message_start",
        "message": {
            **message,
            "content": [],
            "stop_reason": None,
            "usage": {
                "input_tokens": message["usage"]["input_tokens"],
                "output_tokens": 0,
            },
        },
    }

    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            start: Json = {"type": "text", "text": ""}
            delta: Json = {"type": "text_delta", "text": block["text"]}
        else:
            start = {**block, "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"]),
            }

        yield {"type": "content_block_start", "index": index, "content_block": start}
        yield {"type": "content_block_delta", "index": index, "delta": delta}
        yield {"type": "content_block_stop", "index": index}

    yield {
        "type": "message_delta",
        "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
        "usage": {"output_tokens": message["usage"]["output_tokens"]},
    }

    yield {"type": "message_stop"}


def encode_sse(events: Iterable[Json]) -> Iterator[bytes]:
    for event in events:
        data = json.dumps(event, separators=(",", ":"))
        yield f"event: {event['type']}\ndata: {data}\n\n".encode()
