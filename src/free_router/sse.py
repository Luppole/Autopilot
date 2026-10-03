"""Incremental parsing of server-sent events from raw byte chunks."""

import json
from typing import Any


class SSEParser:
    """Feed raw chunks; get back the `data` payload of each completed
    event. Comments, `event:` and `id:` lines are ignored."""

    def __init__(self) -> None:
        self._buffer = b""
        self._data: list[str] = []

    def feed(self, chunk: bytes) -> list[str]:
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split(b"\n")
        return [event for line in lines if (event := self._line(line)) is not None]

    def finish(self) -> list[str]:
        """Flush a final event that wasn't followed by a blank line."""

        events = self.feed(b"\n") if self._buffer else []

        if self._data:
            events.append("\n".join(self._data))
            self._data = []

        return events

    def _line(self, raw: bytes) -> str | None:
        line = raw.rstrip(b"\r").decode("utf-8", errors="replace")

        if not line:
            if not self._data:
                return None
            event = "\n".join(self._data)
            self._data = []
            return event

        if line.startswith("data:"):
            value = line[5:]
            self._data.append(value[1:] if value.startswith(" ") else value)

        return None


def parse_chunk(data: str) -> dict[str, Any] | None:
    """An OpenAI stream chunk as a dict; None for `[DONE]` or junk."""

    if data.strip() == "[DONE]":
        return None

    try:
        chunk = json.loads(data)
    except ValueError:
        return None

    return chunk if isinstance(chunk, dict) else None
