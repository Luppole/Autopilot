import json

from free_router.routing import RouteResult


def routing_headers(result: RouteResult) -> dict[str, str]:
    """Expose which model answered and which ones were skipped, so
    clients (like the dashboard) can show how a request was routed."""

    return {
        "X-Router-Model": result.model,
        "X-Router-Attempts": json.dumps(
            [
                {"model": attempt.model, "status": attempt.status}
                for attempt in result.attempts
            ]
        ),
    }
