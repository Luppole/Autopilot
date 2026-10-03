from fastapi import FastAPI, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from free_router import __version__
from free_router.api import anthropic, dashboard, openai
from free_router.api.errors import error_response, register_error_handlers
from free_router.config import APP_TITLE, WEB_DIR, settings


class RequireJSONMiddleware:
    """Reject non-JSON POSTs.

    Browsers can send `text/plain` or form POSTs cross-site without a
    CORS preflight; requiring JSON means another website can't drive
    this server (and your OpenRouter key) from a visitor's browser.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST":
            request = Request(scope)
            content_type = request.headers.get("content-type", "")

            if not content_type.startswith("application/json"):
                response = error_response(
                    request,
                    415,
                    {
                        "message": "Content-Type must be application/json.",
                        "type": "invalid_request_error",
                    },
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)


def create_app() -> FastAPI:
    app = FastAPI(title=APP_TITLE, version=__version__)

    register_error_handlers(app)

    # Middleware added last runs first: unknown hosts (DNS rebinding)
    # are dropped before anything else sees the request.
    app.add_middleware(RequireJSONMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    app.include_router(dashboard.router)
    app.include_router(openai.router)
    app.include_router(anthropic.router)

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    return app


app = create_app()
