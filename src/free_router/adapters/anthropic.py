"""Anthropic Messages API <-> OpenAI chat completions.

Input is client-supplied JSON, so every helper tolerates missing or
oddly-typed fields instead of trusting the shape.
"""

import json
import time
import uuid
from collections.abc import Iterable, Iterator
from typing import Any

from free_router.errors import InvalidRequestError
from free_router.sse import SSEParser, parse_chunk

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
        # Some providers report "stop" even when they called a tool.
        "stop_reason": (
            "tool_use"
            if any(block["type"] == "tool_use" for block in content)
            else STOP_REASONS.get(choice.get("finish_reason"), "end_turn")
        ),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


# ============================================================
# Streaming: OpenAI chunks -> Anthropic SSE events, as they arrive
# ============================================================

# Send a `ping` after this many quiet seconds (e.g. while a model
# reasons, or a long tool call is buffered) so clients don't time out.
PING_INTERVAL = 10.0


def stream_error_event(message: str) -> Json:
    return {"type": "error", "error": {"type": "api_error", "message": message}}


class _StreamTranslator:
    """Text is passed on as it arrives. Each tool call is held until
    it's complete and then sent whole, with its arguments parsed, so
    providers' quirks in streaming tool calls can't produce bad JSON."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.index = 0
        self.text_open = False
        self.tools: dict[Any, Json] = {}
        self.last_tool_key: Any = 0
        self.used_tools = False
        self.stop_reason = "end_turn"
        self.usage: Json = {}

    def start(self) -> Json:
        return {
            "type": "message_start",
            "message": {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }

    def chunk(self, chunk: Json) -> Iterator[Json]:
        usage = chunk.get("usage")

        if isinstance(usage, dict):
            self.usage = usage

        choices = chunk.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None

        if not isinstance(choice, dict):
            return

        delta = choice.get("delta")

        if isinstance(delta, dict):
            text = _message_text(delta.get("content"))

            if text:
                yield from self._text(text)

            for tool_call in _blocks(delta.get("tool_calls")):
                yield from self._close_text()
                self._tool_fragment(tool_call)

        finish_reason = choice.get("finish_reason")

        if finish_reason:
            self.stop_reason = STOP_REASONS.get(finish_reason, "end_turn")

    def finish(self) -> Iterator[Json]:
        yield from self._close_text()
        yield from self._flush_tools()

        # Anthropic clients expect at least one block.
        if self.index == 0:
            yield from self._text("")
            yield from self._close_text()

        yield {
            "type": "message_delta",
            "delta": {
                "stop_reason": "tool_use" if self.used_tools else self.stop_reason,
                "stop_sequence": None,
            },
            "usage": {
                "input_tokens": int(self.usage.get("prompt_tokens") or 0),
                "output_tokens": int(self.usage.get("completion_tokens") or 0),
            },
        }
        yield {"type": "message_stop"}

    def _text(self, text: str) -> Iterator[Json]:
        # Text after tool calls means those calls are complete.
        yield from self._flush_tools()

        if not self.text_open:
            self.text_open = True
            yield {
                "type": "content_block_start",
                "index": self.index,
                "content_block": {"type": "text", "text": ""},
            }

        if text:
            yield {
                "type": "content_block_delta",
                "index": self.index,
                "delta": {"type": "text_delta", "text": text},
            }

    def _close_text(self) -> Iterator[Json]:
        if self.text_open:
            self.text_open = False
            yield {"type": "content_block_stop", "index": self.index}
            self.index += 1

    def _tool_fragment(self, tool_call: Json) -> None:
        # Calls are told apart by `index`; some providers only send `id`,
        # and fragments with neither continue the previous call.
        key = tool_call.get("index")

        if key is None:
            key = tool_call.get("id") or self.last_tool_key

        self.last_tool_key = key
        tool = self.tools.setdefault(key, {"id": None, "name": "", "arguments": ""})

        if tool_call.get("id") and not tool["id"]:
            tool["id"] = str(tool_call["id"])

        function = tool_call.get("function")

        if not isinstance(function, dict):
            return

        if function.get("name") and not tool["name"]:
            tool["name"] = str(function["name"])

        arguments = function.get("arguments")

        if isinstance(arguments, dict):
            tool["arguments"] = json.dumps(arguments)
        elif isinstance(arguments, str):
            tool["arguments"] += arguments

    def _flush_tools(self) -> Iterator[Json]:
        for tool in self.tools.values():
            block = {
                "type": "tool_use",
                "id": tool["id"] or f"toolu_{uuid.uuid4().hex}",
                "name": tool["name"],
                "input": {},
            }
            arguments = json.dumps(_parse_arguments(tool["arguments"]))

            yield {
                "type": "content_block_start",
                "index": self.index,
                "content_block": block,
            }
            yield {
                "type": "content_block_delta",
                "index": self.index,
                "delta": {"type": "input_json_delta", "partial_json": arguments},
            }
            yield {"type": "content_block_stop", "index": self.index}
            self.index += 1
            self.used_tools = True

        self.tools.clear()


def stream_to_anthropic(chunks: Iterable[bytes], model: str) -> Iterator[Json]:
    """Translate a raw OpenAI SSE byte stream into Anthropic events."""

    translator = _StreamTranslator(model)
    parser = SSEParser()
    last_sent = time.monotonic()

    yield translator.start()

    for raw in chunks:
        events: list[Json] = []

        for data in parser.feed(raw):
            if data.strip() == "[DONE]":
                yield from events
                yield from translator.finish()
                return

            chunk = parse_chunk(data)

            if chunk is None:
                continue

            error = chunk.get("error")

            if error is not None:
                message = error.get("message") if isinstance(error, dict) else error
                yield from events
                yield stream_error_event(f"Upstream error: {message}")
                return

            events.extend(translator.chunk(chunk))

        now = time.monotonic()

        if events:
            yield from events
            last_sent = now
        elif now - last_sent >= PING_INTERVAL:
            yield {"type": "ping"}
            last_sent = now

    for data in parser.finish():
        chunk = parse_chunk(data)

        if chunk is not None:
            yield from translator.chunk(chunk)

    yield from translator.finish()


def encode_sse(events: Iterable[Json]) -> Iterator[bytes]:
    for event in events:
        data = json.dumps(event, separators=(",", ":"))
        yield f"event: {event['type']}\ndata: {data}\n\n".encode()
