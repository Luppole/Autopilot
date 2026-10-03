"""Fallback routing: try free models in random order until one answers."""

import logging
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import requests

from free_router.config import settings
from free_router.errors import InvalidRequestError, NoModelsError, RouterError
from free_router.openrouter import post_chat
from free_router.storage import load_free_models

log = logging.getLogger(__name__)

Payload = dict[str, Any]

# Upstream error bodies can be whole HTML pages; keep reports readable.
MAX_ERROR_LENGTH = 500


@dataclass(frozen=True)
class Attempt:
    model: str
    status: int | None
    error: str


@dataclass
class RouteResult:
    model: str
    attempts: list[Attempt] = field(default_factory=list)
    data: Payload | None = None
    response: requests.Response | None = None


class AllModelsFailed(RouterError):
    status_code = 503
    error_type = "all_models_failed"

    def __init__(self, attempts: list[Attempt]) -> None:
        self.attempts = attempts
        last_error = attempts[-1].error if attempts else "no models were tried"
        super().__init__(
            f"All {len(attempts)} free models failed. Last error: {last_error}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "attempts": [asdict(attempt) for attempt in self.attempts],
        }


class _Miss(Exception):
    """One model didn't produce a usable answer; try the next."""

    def __init__(self, attempt: Attempt) -> None:
        super().__init__(attempt.error)
        self.attempt = attempt


def _truncate(text: str) -> str:
    if len(text) <= MAX_ERROR_LENGTH:
        return text
    return text[:MAX_ERROR_LENGTH] + "…"


def candidate_models(requested: str | None) -> list[str]:
    """The models to try, in order.

    "auto" (or nothing) means every free model, shuffled so the same
    one isn't hammered every time. An explicit id is tried alone, and
    must be a known free model so a typo can't bill a paid one.
    """

    free_ids = [model["id"] for model in load_free_models()]

    if not free_ids:
        raise NoModelsError("No free models found. Run `free-router sync` first.")

    if requested and requested != "auto":
        if requested not in free_ids:
            raise InvalidRequestError(
                f'{requested!r} is not a known free model. Use "auto" '
                "or an id from GET /v1/models."
            )
        return [requested]

    random.shuffle(free_ids)

    if settings.max_attempts:
        return free_ids[: settings.max_attempts]

    return free_ids


def _open(model_id: str, payload: Payload, stream: bool) -> requests.Response:
    try:
        response = post_chat({**payload, "model": model_id}, stream=stream)
    except requests.RequestException as error:
        raise _Miss(Attempt(model_id, None, _truncate(str(error)))) from error

    if not response.ok:
        attempt = Attempt(model_id, response.status_code, _truncate(response.text))
        response.close()
        raise _Miss(attempt)

    return response


def _parse(model_id: str, response: requests.Response) -> Payload:
    try:
        data = response.json()
    except ValueError:
        raise _Miss(
            Attempt(model_id, response.status_code, "Response was not valid JSON")
        ) from None

    # Some providers answer 200 with an error body and no choices.
    choices = data.get("choices") if isinstance(data, dict) else None

    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _Miss(
            Attempt(
                model_id,
                response.status_code,
                _truncate(f"Response had no choices: {data}"),
            )
        )

    payload: Payload = data
    payload["model"] = model_id
    return payload


def route(payload: Payload, stream: bool = False) -> RouteResult:
    """Send `payload` to the first free model that accepts it.

    Non-streaming results carry the parsed JSON in `data`; streaming
    results carry the open `response`, which the caller must close.
    """

    messages = payload.get("messages")

    if not isinstance(messages, list) or not messages:
        raise InvalidRequestError("`messages` must be a non-empty list.")

    payload = {**payload, "stream": stream}
    attempts: list[Attempt] = []

    for model_id in candidate_models(payload.get("model")):
        try:
            response = _open(model_id, payload, stream)
            data = None if stream else _parse(model_id, response)
        except _Miss as miss:
            log.info("%s skipped: %s", model_id, miss.attempt.status or miss)
            attempts.append(miss.attempt)

            if miss.attempt.status == 429:
                time.sleep(settings.rate_limit_delay)

            continue

        log.info("%s answered after %d skipped", model_id, len(attempts))

        if stream:
            return RouteResult(model_id, attempts, response=response)

        return RouteResult(model_id, attempts, data=data)

    raise AllModelsFailed(attempts)
