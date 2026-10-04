"""Small file helpers shared by the workflow stages: atomic writes, CSV/JSON and checksums."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

Row = dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` so that readers never observe a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Row], fieldnames: Sequence[str] | None = None) -> None:
    """Write ``rows``; columns default to the sorted union of all keys. An empty table yields an empty file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames) if fieldnames is not None else sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        if columns:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)
