"""Keeps `uvicorn router:app` working.

Prefer `python -m free_router serve`.
"""

from free_router.api.app import app  # noqa: F401
