import logging
from collections.abc import Iterator
from typing import Any

import requests
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse, StreamingResponse

from free_router.api.headers import routing_headers
from free_router.routing import route
from free_router.storage import load_free_models

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": model["id"], "object": "model", "owned_by": "openrouter"}
            for model in load_free_models()
        ],
    }


# Handlers are plain `def` so FastAPI runs them in its threadpool and
# the blocking upstream calls don't stall the event loop.
@router.post("/v1/chat/completions", response_model=None)
def chat_completions(
    body: dict[str, Any] = Body(...),
) -> JSONResponse | StreamingResponse:
    if body.get("stream"):
        return stream_completion(body)

    result = route(body)

    return JSONResponse(result.data, headers=routing_headers(result))


def stream_completion(body: dict[str, Any]) -> StreamingResponse:
    result = route(body, stream=True)
    response, chunks = result.response, result.chunks
    assert response is not None and chunks is not None

    def relay() -> Iterator[bytes]:
        # Pass bytes through untouched so multi-line SSE events and
        # keep-alive comments arrive exactly as OpenRouter sent them.
        try:
            yield from chunks
        except requests.RequestException as error:
            log.warning("Stream from %s broke off: %s", result.model, error)
        finally:
            response.close()

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers=routing_headers(result),
    )
