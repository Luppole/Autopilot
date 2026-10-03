"""Thin HTTP client for the OpenRouter API."""

from typing import Any

import requests

from free_router.config import (
    APP_TITLE,
    OPENROUTER_CHAT_URL,
    OPENROUTER_MODELS_URL,
    get_api_key,
    settings,
)
from free_router.errors import UpstreamError

# One pooled session reuses TLS connections across requests.
_session = requests.Session()


def fetch_models() -> list[dict[str, Any]]:
    try:
        response = _session.get(OPENROUTER_MODELS_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as error:
        raise UpstreamError(
            f"Couldn't fetch the model list from OpenRouter: {error}"
        ) from error

    models = data.get("data") if isinstance(data, dict) else None

    if not isinstance(models, list):
        raise UpstreamError("OpenRouter returned an unexpected model list.")

    return [model for model in models if isinstance(model, dict)]


def post_chat(payload: dict[str, Any], timeout: float) -> requests.Response:
    """Send a chat request. The body is left unread, so the caller can
    stream it or read it under its own deadline, and must close it."""

    return _session.post(
        OPENROUTER_CHAT_URL,
        headers={
            "Authorization": f"Bearer {get_api_key()}",
            "X-Title": APP_TITLE,
        },
        json=payload,
        timeout=(min(settings.connect_timeout, timeout), timeout),
        stream=True,
    )
