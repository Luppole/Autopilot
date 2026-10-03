import logging
from collections.abc import Iterator
from typing import Any

import requests
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse, StreamingResponse

from free_router.adapters.anthropic import (
    encode_sse,
    request_to_openai,
    response_to_anthropic,
    stream_error_event,
    stream_to_anthropic,
)
from free_router.api.headers import routing_headers
from free_router.routing import route

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/messages", response_model=None)
def messages(body: dict[str, Any] = Body(...)) -> JSONResponse | StreamingResponse:
    payload = request_to_openai(body)

    if body.get("stream"):
        return stream_messages(payload)

    result = route(payload)
    assert result.data is not None

    return JSONResponse(
        response_to_anthropic(result.data, result.model),
        headers=routing_headers(result),
    )


def stream_messages(payload: dict[str, Any]) -> StreamingResponse:
    # Routing reads until the model's first output, so a model that
    # fails early still falls back to the next one before we commit.
    result = route(payload, stream=True)
    response, chunks = result.response, result.chunks
    assert response is not None and chunks is not None

    def relay() -> Iterator[bytes]:
        try:
            yield from encode_sse(stream_to_anthropic(chunks, result.model))
        except requests.RequestException as error:
            log.warning("Stream from %s broke off: %s", result.model, error)
            yield from encode_sse(
                [stream_error_event("The upstream stream broke off.")]
            )
        finally:
            response.close()

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers=routing_headers(result),
    )
