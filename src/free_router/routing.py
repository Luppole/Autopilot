"""Fallback routing: try capable free models, healthiest first, until one answers."""

import itertools
import json
import logging
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import requests

from free_router import health
from free_router.capabilities import estimate_prompt_tokens, fit_payload, unfit_reason
from free_router.config import settings
from free_router.errors import InvalidRequestError, NoModelsError, RouterError
from free_router.openrouter import post_chat
from free_router.sse import SSEParser, parse_chunk
from free_router.storage import Record, load_free_models

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
    # The stream's raw bytes, starting from the beginning.
    chunks: Iterator[bytes] | None = None


class AllModelsFailed(RouterError):
    status_code = 503
    error_type = "all_models_failed"

    def __init__(self, attempts: list[Attempt]) -> None:
        self.attempts = attempts
        super().__init__(
            f"All {len(attempts)} free models failed. {_last_error(attempts)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "attempts": [asdict(attempt) for attempt in self.attempts],
        }


class DeadlineExceeded(AllModelsFailed):
    status_code = 504
    error_type = "deadline_exceeded"

    def __init__(self, attempts: list[Attempt], seconds: float) -> None:
        super().__init__(attempts)
        self.message = (
            f"No free model answered within {seconds:g}s "
            f"({len(attempts)} tried). {_last_error(attempts)}"
        )
        self.args = (self.message,)


def _last_error(attempts: list[Attempt]) -> str:
    if not attempts:
        return "No models were tried."
    return f"Last error: {attempts[-1].error}"


class _Miss(Exception):
    """One model didn't produce a usable answer; try the next."""

    def __init__(self, attempt: Attempt) -> None:
        super().__init__(attempt.error)
        self.attempt = attempt


def _truncate(text: str) -> str:
    if len(text) <= MAX_ERROR_LENGTH:
        return text
    return text[:MAX_ERROR_LENGTH] + "…"


def candidate_models(payload: Payload, prompt_tokens: int) -> list[Record]:
    """The models to try, in order.

    "auto" (or nothing) means every free model that can handle the
    request, healthiest first (see `health.order`). An explicit id is
    tried alone, and must be a known free model so a typo can't bill
    a paid one.
    """

    free_models = {model["id"]: model for model in load_free_models()}

    if not free_models:
        raise NoModelsError("No free models found. Run `free-router sync` first.")

    requested = payload.get("model")

    if requested and requested != "auto":
        if requested not in free_models:
            raise InvalidRequestError(
                f'{requested!r} is not a known free model. Use "auto" '
                "or an id from GET /v1/models."
            )
        return [free_models[requested]]

    reasons = {
        model_id: unfit_reason(model, payload, prompt_tokens)
        for model_id, model in free_models.items()
    }
    fit_ids = [model_id for model_id, reason in reasons.items() if reason is None]

    if not fit_ids:
        details = "; ".join(f"{m} {r}" for m, r in reasons.items())
        raise InvalidRequestError(
            f"No free model can handle this request (about {prompt_tokens} "
            f"prompt tokens): {details}."
        )

    ordered = health.order(fit_ids)

    if settings.max_attempts:
        ordered = ordered[: settings.max_attempts]

    return [free_models[model_id] for model_id in ordered]


def _open(model_id: str, payload: Payload, timeout: float) -> requests.Response:
    try:
        response = post_chat({**payload, "model": model_id}, timeout=timeout)
    except requests.RequestException as error:
        raise _Miss(Attempt(model_id, None, _truncate(str(error)))) from error

    if not response.ok:
        attempt = Attempt(model_id, response.status_code, _truncate(response.text))
        response.close()
        raise _Miss(attempt)

    return response


def _read_body(
    model_id: str, response: requests.Response, deadline: float | None
) -> bytes:
    """Read the whole body, giving up at `deadline`.

    The socket read timeout only bounds gaps between bytes, so a reply
    that trickles in (e.g. keep-alive whitespace) could outlast it.
    """

    chunks: list[bytes] = []

    try:
        for chunk in response.iter_content(chunk_size=None):
            chunks.append(chunk)

            if deadline is not None and time.monotonic() > deadline:
                raise _Miss(
                    Attempt(model_id, None, "Request deadline reached mid-reply")
                )
    except requests.RequestException as error:
        raise _Miss(Attempt(model_id, None, _truncate(str(error)))) from error
    finally:
        response.close()

    return b"".join(chunks)


def _parse(model_id: str, response: requests.Response, body: bytes) -> Payload:
    try:
        data = json.loads(body)
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


def _stream_started(model_id: str, data: str) -> bool:
    """Whether this stream event shows the model is really answering."""

    if data.strip() == "[DONE]":
        raise _Miss(Attempt(model_id, None, "Stream ended before any output"))

    chunk = parse_chunk(data)

    if chunk is None:
        return False

    # Providers can fail after the 200, reporting it as an event.
    error = chunk.get("error")

    if error is not None:
        code = error.get("code") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else error
        raise _Miss(
            Attempt(
                model_id,
                code if isinstance(code, int) else None,
                _truncate(str(message)),
            )
        )

    choices = chunk.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None

    if not isinstance(choice, dict):
        return False

    delta = choice.get("delta")

    if not isinstance(delta, dict):
        delta = {}

    return bool(
        delta.get("content")
        or delta.get("tool_calls")
        or delta.get("reasoning")
        or choice.get("finish_reason")
    )


def _start_stream(
    model_id: str, response: requests.Response, deadline: float | None
) -> Iterator[bytes]:
    """Read until the model produces output, so a stream that errors or
    ends empty can still fall back to the next model. Returns all the
    bytes, including those already read."""

    source = response.iter_content(chunk_size=None)
    seen: list[bytes] = []
    parser = SSEParser()

    try:
        for chunk in source:
            seen.append(chunk)

            if any(_stream_started(model_id, data) for data in parser.feed(chunk)):
                return itertools.chain(seen, source)

            if deadline is not None and time.monotonic() > deadline:
                raise _Miss(
                    Attempt(model_id, None, "Request deadline reached before output")
                )

        if any(_stream_started(model_id, data) for data in parser.finish()):
            return iter(seen)
    except requests.RequestException as error:
        response.close()
        raise _Miss(Attempt(model_id, None, _truncate(str(error)))) from error
    except _Miss:
        response.close()
        raise

    response.close()
    raise _Miss(Attempt(model_id, None, "Stream ended before any output"))


def route(payload: Payload, stream: bool = False) -> RouteResult:
    """Send `payload` to the first free model that accepts it.

    Non-streaming results carry the parsed JSON in `data`. Streaming
    results carry the stream's bytes in `chunks` and the open
    `response`, which the caller must close.
    """

    messages = payload.get("messages")

    if not isinstance(messages, list) or not messages:
        raise InvalidRequestError("`messages` must be a non-empty list.")

    payload = {**payload, "stream": stream}
    prompt_tokens = estimate_prompt_tokens(payload)
    attempts: list[Attempt] = []
    deadline = (
        time.monotonic() + settings.request_deadline
        if settings.request_deadline
        else None
    )

    for model in candidate_models(payload, prompt_tokens):
        model_id = model["id"]
        started = time.monotonic()
        timeout = settings.request_timeout

        if deadline is not None:
            if started >= deadline:
                raise DeadlineExceeded(attempts, settings.request_deadline)
            timeout = min(timeout, deadline - started)

        try:
            response = _open(
                model_id, fit_payload(model, payload, prompt_tokens), timeout
            )
            if stream:
                chunks = _start_stream(model_id, response, deadline)
            else:
                data = _parse(
                    model_id, response, _read_body(model_id, response, deadline)
                )
        except _Miss as miss:
            log.info("%s skipped: %s", model_id, miss.attempt.status or miss)
            attempts.append(miss.attempt)
            out_of_time = deadline is not None and time.monotonic() >= deadline

            # A model cut short by our deadline didn't necessarily fail.
            if health.is_model_failure(miss.attempt.status) and not out_of_time:
                health.record_failure(model_id)

            if miss.attempt.status == 429 and not out_of_time:
                time.sleep(settings.rate_limit_delay)

            continue

        health.record_success(model_id, time.monotonic() - started)
        log.info("%s answered after %d skipped", model_id, len(attempts))

        if stream:
            return RouteResult(model_id, attempts, response=response, chunks=chunks)

        return RouteResult(model_id, attempts, data=data)

    if deadline is not None and time.monotonic() >= deadline:
        raise DeadlineExceeded(attempts, settings.request_deadline)

    raise AllModelsFailed(attempts)
