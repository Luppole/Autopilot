import threading
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from free_router import autosync
from free_router.api.app import app
from free_router.autosync import AutoSync, last_synced_at, seconds_until_due
from free_router.config import settings
from free_router.errors import UpstreamError
from free_router.storage import write_json
from free_router.sync import SyncResult

NOW = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)


def synced(*times: str) -> None:
    write_json(settings.history_file, [{"synced_at": t} for t in times])


class FakeSync:
    """Counts calls and lets a test wait for the Nth one."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0
        self.called = threading.Event()

    def __call__(self) -> SyncResult:
        self.calls += 1
        self.called.set()

        if self.error:
            raise self.error

        return SyncResult(models=[], added=[], removed=[], synced_at="")


@pytest.fixture
def fake_sync(monkeypatch: pytest.MonkeyPatch) -> FakeSync:
    fake = FakeSync()
    monkeypatch.setattr(autosync, "sync_free_models", fake)
    return fake


def test_last_synced_at_uses_newest_valid_entry() -> None:
    synced("2026-10-01T00:00:00+00:00", "2026-10-03T00:00:00", "garbage")

    assert last_synced_at() == datetime(2026, 10, 3, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        ([], 0),
        (["2026-10-04T11:00:00+00:00"], 5 * 3600),
        (["2026-10-03T00:00:00+00:00"], 0),
    ],
)
def test_seconds_until_due(history: list[str], expected: float) -> None:
    synced(*history)

    assert seconds_until_due(6 * 3600, now=NOW) == expected


def test_syncs_at_start_when_stale(fake_sync: FakeSync) -> None:
    auto = AutoSync(interval=3600)
    auto.start()

    try:
        assert fake_sync.called.wait(timeout=5)
    finally:
        auto.stop()


def test_waits_when_recently_synced(fake_sync: FakeSync) -> None:
    synced(datetime.now(timezone.utc).isoformat())
    auto = AutoSync(interval=3600)
    auto.start()
    auto.stop()

    assert fake_sync.calls == 0


def test_disabled_with_zero_interval(fake_sync: FakeSync) -> None:
    auto = AutoSync(interval=0)
    auto.start()
    auto.stop()

    assert fake_sync.calls == 0


def test_failure_is_logged_and_retried_soon(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        autosync, "sync_free_models", FakeSync(UpstreamError("offline"))
    )

    delay = AutoSync(interval=6 * 3600).sync_once()

    assert delay == autosync.RETRY_DELAY
    assert "offline" in caplog.text


def test_success_waits_a_full_interval(fake_sync: FakeSync) -> None:
    assert AutoSync(interval=6 * 3600).sync_once() == 6 * 3600


def test_server_starts_and_stops_auto_sync(
    fake_sync: FakeSync, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "sync_interval", 3600)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert fake_sync.called.wait(timeout=5)
        assert client.get("/health").status_code == 200
