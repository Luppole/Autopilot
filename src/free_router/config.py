"""Runtime settings, read once from environment variables.

Modules read `settings` attributes at call time (never copy them at
import), so tests and embedders can adjust them in place.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from free_router.errors import ConfigError

APP_TITLE = "Free Model Router"
WEB_DIR = Path(__file__).resolve().parent / "web"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"

API_KEY_ENV = "OPENROUTER_API_KEY"


def _env_number(name: str, default: float) -> float:
    raw = os.getenv(name)

    if raw is None or raw.strip() == "":
        return default

    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None

    if value < 0:
        raise ConfigError(f"{name} must not be negative, got {raw!r}")

    return value


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)

    if raw is None or raw.strip() == "":
        return default

    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    data_dir: Path = Path("data")
    history_limit: int = 90
    # Seconds to wait on a single model before moving on.
    request_timeout: float = 300
    # How many models to try per request; 0 means all of them.
    max_attempts: int = 0
    # Pause after a 429 so we don't hammer OpenRouter.
    rate_limit_delay: float = 0.5
    # Host headers the server answers to (guards against DNS rebinding).
    allowed_hosts: list[str] = field(default_factory=lambda: ["127.0.0.1", "localhost"])

    @property
    def models_file(self) -> Path:
        return self.data_dir / "free-models.json"

    @property
    def history_file(self) -> Path:
        return self.data_dir / "free-model-history.json"

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()

        return cls(
            data_dir=Path(os.getenv("FREE_ROUTER_DATA_DIR") or defaults.data_dir),
            history_limit=int(
                _env_number("FREE_ROUTER_HISTORY_LIMIT", defaults.history_limit)
            ),
            request_timeout=_env_number(
                "FREE_ROUTER_REQUEST_TIMEOUT", defaults.request_timeout
            ),
            max_attempts=int(
                _env_number("FREE_ROUTER_MAX_ATTEMPTS", defaults.max_attempts)
            ),
            rate_limit_delay=_env_number(
                "FREE_ROUTER_RATE_LIMIT_DELAY", defaults.rate_limit_delay
            ),
            allowed_hosts=_env_list(
                "FREE_ROUTER_ALLOWED_HOSTS", defaults.allowed_hosts
            ),
        )


def get_api_key() -> str:
    api_key = os.getenv(API_KEY_ENV, "").strip()

    if not api_key:
        raise ConfigError(
            f"{API_KEY_ENV} is not set. Create a key at "
            "https://openrouter.ai/keys and export it before starting."
        )

    return api_key


settings = Settings.from_env()
