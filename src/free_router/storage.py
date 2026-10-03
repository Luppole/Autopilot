"""JSON files on disk: the current free model list and the sync log."""

import contextlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from free_router.config import settings

log = logging.getLogger(__name__)

Record = dict[str, Any]


def read_json(path: Path, fallback: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return fallback
    except (OSError, json.JSONDecodeError) as error:
        log.warning("Ignoring unreadable %s: %s", path, error)
        return fallback


def write_json(path: Path, data: Any) -> None:
    """Write atomically, so readers never see a half-written file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")

        _replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def _replace(source: str, target: Path, retries: int = 5) -> None:
    # On Windows the swap fails while another thread has the target
    # open for reading; the reader is quick, so retry briefly.
    for attempt in range(retries):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _read_records(path: Path) -> list[Record]:
    data = read_json(path, [])

    if not isinstance(data, list):
        log.warning("Ignoring %s: expected a JSON list", path)
        return []

    return [item for item in data if isinstance(item, dict)]


def load_free_models() -> list[Record]:
    return [
        model
        for model in _read_records(settings.models_file)
        if isinstance(model.get("id"), str)
    ]


def load_history() -> list[Record]:
    return _read_records(settings.history_file)
