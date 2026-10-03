"""Collect the current list of free OpenRouter models and track changes."""

import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from free_router.config import settings
from free_router.openrouter import fetch_models
from free_router.storage import Record, load_free_models, load_history, write_json

# Serializes syncs so two at once can't interleave their writes.
_lock = threading.Lock()


@dataclass(frozen=True)
class SyncResult:
    models: list[Record]
    added: list[Record]
    removed: list[Record]
    synced_at: str

    def summary(self) -> dict[str, Any]:
        return {
            "synced_at": self.synced_at,
            "total_free_models": len(self.models),
            "added": [model["id"] for model in self.added],
            "removed": [model["id"] for model in self.removed],
        }


def _price(pricing: dict[str, Any], key: str) -> float:
    try:
        return float(pricing.get(key, math.inf))
    except (TypeError, ValueError):
        return math.inf


def is_free_model(model: Record) -> bool:
    model_id = model.get("id")
    pricing = model.get("pricing")

    return (
        isinstance(model_id, str)
        and model_id.endswith(":free")
        and isinstance(pricing, dict)
        and _price(pricing, "prompt") == 0
        and _price(pricing, "completion") == 0
    )


def normalize_model(model: Record, collected_at: str) -> Record:
    return {
        "id": model["id"],
        "name": model.get("name"),
        "description": model.get("description"),
        "context_length": model.get("context_length"),
        "architecture": model.get("architecture"),
        "pricing": model.get("pricing"),
        # Used to skip models that can't handle a request (see capabilities).
        "supported_parameters": model.get("supported_parameters"),
        "top_provider": model.get("top_provider"),
        "created": model.get("created"),
        "updated": model.get("updated"),
        "collected_at": collected_at,
    }


def sync_free_models() -> SyncResult:
    # Fetch outside the lock: it's the slow part and touches no files.
    fetched = fetch_models()

    with _lock:
        now = datetime.now(timezone.utc)
        synced_at = now.isoformat()

        free_models = sorted(
            (normalize_model(m, synced_at) for m in fetched if is_free_model(m)),
            key=lambda model: model["id"],
        )

        previous = load_free_models()
        previous_ids = {model["id"] for model in previous}
        current_ids = {model["id"] for model in free_models}

        result = SyncResult(
            models=free_models,
            added=[m for m in free_models if m["id"] not in previous_ids],
            removed=[m for m in previous if m["id"] not in current_ids],
            synced_at=synced_at,
        )

        history = load_history()
        history.append({"date": now.strftime("%Y-%m-%d"), **result.summary()})

        write_json(settings.models_file, free_models)
        write_json(settings.history_file, history[-settings.history_limit :])

    return result
