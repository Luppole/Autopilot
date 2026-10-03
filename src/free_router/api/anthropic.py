from typing import Any

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse, StreamingResponse

from free_router.adapters.anthropic import (
    encode_sse,
    request_to_openai,
    response_to_anthropic,
    stream_events,
)
from free_router.api.headers import routing_headers
from free_router.routing import route

router = APIRouter()


@router.post("/v1/messages", response_model=None)
def messages(body: dict[str, Any] = Body(...)) -> JSONResponse | StreamingResponse:
    # Always fetch the full answer, then replay it as SSE if asked:
    # this keeps tool calls intact regardless of the provider's
    # streaming quirks.
    result = route(request_to_openai(body))
    assert result.data is not None

    message = response_to_anthropic(result.data, result.model)
    headers = routing_headers(result)

    if body.get("stream"):
        return StreamingResponse(
            encode_sse(stream_events(message)),
            media_type="text/event-stream",
            headers=headers,
        )

    return JSONResponse(message, headers=headers)
