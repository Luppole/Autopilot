"""Keep the free model list fresh while the server runs."""

import logging
import threading
from datetime import datetime, timezone

from free_router.errors import RouterError
from free_router.storage import load_history
from free_router.sync import sync_free_models

log = logging.getLogger(__name__)

# After a failed sync (e.g. offline), try again this soon.
RETRY_DELAY = 300.0


def last_synced_at() -> datetime | None:
    for entry in reversed(load_history()):
        value = entry.get("synced_at")

        if not isinstance(value, str):
            continue

        try:
            synced_at = datetime.fromisoformat(value)
        except ValueError:
            continue

        if synced_at.tzinfo is None:
            synced_at = synced_at.replace(tzinfo=timezone.utc)

        return synced_at

    return None


def seconds_until_due(interval: float, now: datetime | None = None) -> float:
    last = last_synced_at()

    if last is None:
        return 0.0

    elapsed = ((now or datetime.now(timezone.utc)) - last).total_seconds()
    return max(interval - elapsed, 0.0)


class AutoSync:
    """Syncs in a background thread: at startup if the list is stale,
    then every `interval` seconds. Manual syncs push the next one back."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.interval <= 0 or self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run, name="free-router-autosync", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _run(self) -> None:
        delay = seconds_until_due(self.interval)

        while not self._stop.wait(delay):
            # Someone may have synced (dashboard, CLI) while we waited.
            delay = seconds_until_due(self.interval)

            if delay <= 0:
                delay = self.sync_once()

    def sync_once(self) -> float:
        """Sync now; return how long to wait before the next one."""

        try:
            result = sync_free_models()
        except RouterError as error:
            log.warning("Auto-sync failed, retrying soon: %s", error.message)
            return min(RETRY_DELAY, self.interval)
        except Exception:
            log.exception("Auto-sync failed unexpectedly")
            return min(RETRY_DELAY, self.interval)

        log.info(
            "Auto-sync: %d free models (+%d, -%d)",
            len(result.models),
            len(result.added),
            len(result.removed),
        )
        return self.interval
