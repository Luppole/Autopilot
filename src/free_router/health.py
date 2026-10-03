"""In-memory model health: who's been failing lately, who's reliable.

Lives for the life of the process. Failing models sit out a cooldown
that doubles with each consecutive failure; reliable models are
tried earlier, but with enough randomness that load still spreads.
"""

import random
import threading
import time
from dataclasses import dataclass

# Cooldown after the first consecutive failure, doubling up to the cap.
BASE_COOLDOWN = 10.0
MAX_COOLDOWN = 300.0

# How fast the reliability score follows recent results (0..1).
SCORE_WEIGHT = 0.2
INITIAL_SCORE = 0.5
# Even an unreliable model keeps some chance of being tried early.
MIN_SCORE = 0.05

# Statuses that say something about the model, not about the request.
# Other 4xx (bad params, prompt too long) are the request's fault.
MODEL_FAILURE_STATUSES = {408, 429}


@dataclass
class ModelHealth:
    score: float = INITIAL_SCORE
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    successes: int = 0
    failures: int = 0
    latency: float | None = None


_lock = threading.Lock()
_models: dict[str, ModelHealth] = {}


def is_model_failure(status: int | None) -> bool:
    """Network errors, 5xx, 408/429 and unusable 2xx bodies count
    against a model; other 4xx are blamed on the request."""

    if status is None or status in MODEL_FAILURE_STATUSES:
        return True

    return not 400 <= status < 500


def record_success(model_id: str, latency: float) -> None:
    with _lock:
        health = _models.setdefault(model_id, ModelHealth())
        health.score += SCORE_WEIGHT * (1 - health.score)
        health.consecutive_failures = 0
        health.cooldown_until = 0.0
        health.successes += 1
        health.latency = (
            latency
            if health.latency is None
            else health.latency + SCORE_WEIGHT * (latency - health.latency)
        )


def record_failure(model_id: str) -> None:
    with _lock:
        health = _models.setdefault(model_id, ModelHealth())
        health.score -= SCORE_WEIGHT * health.score
        health.consecutive_failures += 1
        health.failures += 1
        cooldown = BASE_COOLDOWN * 2 ** (health.consecutive_failures - 1)
        health.cooldown_until = time.monotonic() + min(cooldown, MAX_COOLDOWN)


def order(model_ids: list[str]) -> list[str]:
    """Available models first, as a reliability-weighted shuffle; models
    cooling down go last, soonest-available first, as a last resort."""

    now = time.monotonic()

    with _lock:
        snapshot = {
            model_id: _models.get(model_id, ModelHealth()) for model_id in model_ids
        }

    available = [m for m in model_ids if snapshot[m].cooldown_until <= now]
    cooling = [m for m in model_ids if snapshot[m].cooldown_until > now]

    # Weighted shuffle (Efraimidis-Spirakis): higher score, earlier slot.
    def key(model_id: str) -> float:
        weight = max(snapshot[model_id].score, MIN_SCORE)
        return float(random.random() ** (1 / weight))

    available.sort(key=key, reverse=True)
    cooling.sort(key=lambda m: snapshot[m].cooldown_until)

    return available + cooling


def snapshot() -> dict[str, dict[str, float | int | None]]:
    """Current stats per model, for the API."""

    now = time.monotonic()

    with _lock:
        return {
            model_id: {
                "score": round(health.score, 3),
                "successes": health.successes,
                "failures": health.failures,
                "latency_seconds": (
                    None if health.latency is None else round(health.latency, 2)
                ),
                "cooldown_seconds": round(max(health.cooldown_until - now, 0), 1),
            }
            for model_id, health in _models.items()
        }


def reset() -> None:
    with _lock:
        _models.clear()
