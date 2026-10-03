"""Match a request to the models that can actually serve it.

Uses the metadata saved at sync time. A model with no metadata for
something (e.g. synced by an older version) is assumed to cope.
"""

import json
from typing import Any

from free_router.storage import Record

Payload = dict[str, Any]

# Rough chars-per-token for English text and code. Only used to rule
# out models whose context is clearly too small, so it needn't be exact.
CHARS_PER_TOKEN = 4

MAX_TOKEN_KEYS = ("max_tokens", "max_completion_tokens")


def estimate_prompt_tokens(payload: Payload) -> int:
    text = json.dumps(
        [payload.get("messages"), payload.get("tools")], ensure_ascii=False
    )
    return len(text) // CHARS_PER_TOKEN


def _uses_images(payload: Payload) -> bool:
    for message in payload.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None

        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in content
        ):
            return True

    return False


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _max_completion_tokens(model: Record) -> int | None:
    top_provider = model.get("top_provider")

    if not isinstance(top_provider, dict):
        return None

    return _int(top_provider.get("max_completion_tokens"))


def unfit_reason(model: Record, payload: Payload, prompt_tokens: int) -> str | None:
    """Why `model` can't serve `payload`, or None if it (probably) can."""

    parameters = model.get("supported_parameters")

    if (
        payload.get("tools")
        and isinstance(parameters, list)
        and "tools" not in parameters
    ):
        return "doesn't support tools"

    architecture = model.get("architecture")
    modalities = (
        architecture.get("input_modalities") if isinstance(architecture, dict) else None
    )

    if (
        isinstance(modalities, list)
        and "image" not in modalities
        and _uses_images(payload)
    ):
        return "doesn't accept images"

    context_length = _int(model.get("context_length"))

    if context_length and prompt_tokens >= context_length:
        return f"context of {context_length} tokens is too small"

    return None


def fit_payload(model: Record, payload: Payload, prompt_tokens: int) -> Payload:
    """Lower the requested output limit to what `model` allows, so a
    client asking for 32k tokens isn't rejected by an 8k model."""

    limits = [_max_completion_tokens(model)]
    context_length = _int(model.get("context_length"))

    if context_length and context_length > prompt_tokens:
        limits.append(context_length - prompt_tokens)

    limit = min((value for value in limits if value), default=None)

    if limit is None:
        return payload

    fitted = dict(payload)

    for key in MAX_TOKEN_KEYS:
        requested = payload.get(key)

        if isinstance(requested, int) and requested > limit:
            fitted[key] = limit

    return fitted
