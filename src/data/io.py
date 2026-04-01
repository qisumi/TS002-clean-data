from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import FIGURES_DIR, LOGS_DIR, STATISTIC_RESULTS_DIR


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_project_directories() -> None:
    ensure_directory(STATISTIC_RESULTS_DIR)
    ensure_directory(FIGURES_DIR)
    ensure_directory(LOGS_DIR)


def _temp_path(path: Path) -> Path:
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    return path.with_name(f".{stem}.{uuid.uuid4().hex}.tmp{suffix}")


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    ensure_directory(path.parent)
    temp_path = _temp_path(path)
    temp_path.write_text(content, encoding=encoding)
    temp_path.replace(path)


def write_markdown(path: Path, content: str) -> None:
    atomic_write_text(path, content, encoding="utf-8")


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_directory(path.parent)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    temp_path = _temp_path(path)
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def write_dataframe_csv(path: Path, frame: pd.DataFrame, index: bool = False) -> None:
    ensure_directory(path.parent)
    temp_path = _temp_path(path)
    frame.to_csv(temp_path, index=index)
    temp_path.replace(path)
