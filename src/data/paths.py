from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def _resolve_dir(default_name: str, *env_names: str) -> Path:
    for env_name in env_names:
        raw_value = os.environ.get(env_name, "").strip()
        if not raw_value:
            continue
        path = Path(raw_value).expanduser()
        return path if path.is_absolute() else ROOT_DIR / path
    return ROOT_DIR / default_name


RAW_DIR = _resolve_dir("rawdata", "RAW_DIR")
STATISTIC_RESULTS_DIR = _resolve_dir("statistic_results", "STATS_DIR", "STATISTIC_RESULTS_DIR")
FIGURES_DIR = _resolve_dir("figures", "FIGURES_DIR")
LOGS_DIR = _resolve_dir("logs", "LOGS_DIR")
REPORTS_DIR = _resolve_dir("reports", "REPORTS_DIR")
