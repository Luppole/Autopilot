from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse

from free_router import __version__
from free_router.config import APP_TITLE, WEB_DIR
from free_router.health import snapshot as health_snapshot
from free_router.storage import Record, load_free_models, load_history
from free_router.sync import sync_free_models

router = APIRouter()


@router.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "free_models": len(load_free_models())}


@router.get("/api/status")
def status() -> dict[str, Any]:
    history = load_history()

    return {
        "name": APP_TITLE,
        "version": __version__,
        "free_models": len(load_free_models()),
        "last_synced_at": history[-1].get("synced_at") if history else None,
        "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/messages"],
    }


@router.get("/api/models")
def models() -> list[Record]:
    return load_free_models()


@router.get("/api/model-health")
def model_health() -> dict[str, Any]:
    return health_snapshot()


@router.get("/api/history")
def history() -> list[Record]:
    return load_history()


@router.post("/api/sync")
def sync() -> dict[str, Any]:
    return sync_free_models().summary()
