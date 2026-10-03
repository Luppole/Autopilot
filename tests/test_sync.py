from typing import Any

import pytest

from free_router import sync
from free_router.config import settings
from free_router.storage import load_free_models, load_history


def model(model_id: str, prompt: Any = "0", completion: Any = "0") -> dict[str, Any]:
    return {"id": model_id, "pricing": {"prompt": prompt, "completion": completion}}


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (model("a/b:free"), True),
        (model("a/b"), False),
        (model("a/b:free", prompt="0.1"), False),
        (model("a/b:free", prompt=None), False),
        (model("a/b:free", completion="abc"), False),
        ({"id": "a/b:free"}, False),
        ({"id": "a/b:free", "pricing": "0"}, False),
        ({"id": 7, "pricing": {"prompt": "0", "completion": "0"}}, False),
    ],
)
def test_is_free_model(candidate: dict[str, Any], expected: bool) -> None:
    assert sync.is_free_model(candidate) is expected


def test_sync_records_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sync,
        "fetch_models",
        lambda: [
            model("acme/alpha:free"),
            model("acme/beta:free"),
            model("acme/delta:free"),
            model("acme/paid", prompt="0.002"),
        ],
    )

    result = sync.sync_free_models()

    assert [m["id"] for m in result.added] == ["acme/delta:free"]
    assert [m["id"] for m in result.removed] == ["acme/gamma:free"]
    assert [m["id"] for m in load_free_models()] == [
        "acme/alpha:free",
        "acme/beta:free",
        "acme/delta:free",
    ]

    entry = load_history()[-1]
    assert entry["added"] == ["acme/delta:free"]
    assert entry["removed"] == ["acme/gamma:free"]
    assert entry["total_free_models"] == 3


def test_history_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sync, "fetch_models", lambda: [model("a/b:free")])
    monkeypatch.setattr(settings, "history_limit", 2)

    for _ in range(3):
        sync.sync_free_models()

    assert len(load_history()) == 2
