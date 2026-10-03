"""Turn exceptions into error bodies each kind of client understands.

`/v1/messages` callers are Anthropic SDKs and expect
`{"type": "error", "error": {...}}`; everything else gets the OpenAI
shape `{"error": {"message": ..., "type": ...}}`.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from free_router.errors import RouterError

log = logging.getLogger(__name__)

ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    503: "overloaded_error",
}


def error_response(
    request: Request, status_code: int, error: dict[str, Any]
) -> JSONResponse:
    if request.url.path.startswith("/v1/messages"):
        body: dict[str, Any] = {
            "type": "error",
            "error": {
                "type": ANTHROPIC_ERROR_TYPES.get(status_code, "api_error"),
                "message": error["message"],
            },
        }
    else:
        body = {"error": error}

    return JSONResponse(body, status_code=status_code)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RouterError)
    async def router_error(request: Request, error: RouterError) -> JSONResponse:
        if error.status_code >= 500:
            log.warning("%s: %s", type(error).__name__, error.message)
        return error_response(request, error.status_code, error.to_dict())

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        return error_response(
            request,
            400,
            {
                "message": "Request body must be a JSON object.",
                "type": "invalid_request_error",
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        return error_response(
            request,
            error.status_code,
            {"message": str(error.detail), "type": "invalid_request_error"},
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return error_response(
            request,
            500,
            {"message": "Internal server error.", "type": "api_error"},
        )
