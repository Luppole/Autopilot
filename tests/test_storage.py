from pathlib import Path

from free_router.config import settings
from free_router.storage import load_free_models, load_history, read_json, write_json


def test_write_is_atomic_and_leaves_no_temp_files(data_dir: Path) -> None:
    path = data_dir / "nested" / "file.json"

    write_json(path, {"a": 1})
    write_json(path, {"a": 2})

    assert read_json(path, None) == {"a": 2}
    assert [p.name for p in path.parent.iterdir()] == ["file.json"]


def test_corrupt_files_fall_back(data_dir: Path) -> None:
    settings.models_file.write_text("{not json", encoding="utf-8")

    assert load_free_models() == []


def test_unexpected_shapes_are_filtered(data_dir: Path) -> None:
    write_json(settings.models_file, [{"id": "ok:free"}, {"name": "no id"}, "junk"])
    write_json(settings.history_file, {"not": "a list"})

    assert load_free_models() == [{"id": "ok:free"}]
    assert load_history() == []
