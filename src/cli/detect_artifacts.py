from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data import (
    ROOT_DIR,
    STATISTIC_RESULTS_DIR,
    default_dataset_argument,
    ensure_project_directories,
    list_dataset_files,
    read_dataset,
    relative_path,
    write_csv,
)


DEFAULT_RULES = {
    "zero_block_min_length": 4,
    "zero_tolerance": 1e-12,
    "flat_run_min_length": 8,
    "flat_tolerance": 1e-12,
    "near_constant_window": 24,
    "near_constant_min_length": 24,
    "near_constant_std_ratio": 0.01,
    "near_constant_min_std": 1e-6,
    "repetition_window": 24,
    "repetition_min_length": 48,
    "repetition_tolerance_ratio": 0.001,
    "repetition_min_abs_tolerance": 1e-6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect rule-based artifact candidates.")
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "artifact_rules.yaml"),
        help="Path to the artifact rule config.",
    )
    parser.add_argument(
        "--datasets",
        default=default_dataset_argument(),
        help="Comma-separated dataset names. Defaults to the core analysis datasets.",
    )
    return parser.parse_args()


def load_rules(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {"defaults": DEFAULT_RULES.copy(), "datasets": {}}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = DEFAULT_RULES.copy()
    defaults.update(loaded.get("defaults", {}))
    return {"defaults": defaults, "datasets": loaded.get("datasets", {})}


def resolve_dataset_rules(dataset_name: str, config: dict[str, Any]) -> dict[str, Any]:
    rules = config["defaults"].copy()
    rules.update(config["datasets"].get(dataset_name, {}))
    return rules


def detect_artifacts() -> None:
    args = parse_args()
    ensure_project_directories()
    config = load_rules(args.config)
    dataset_filter = {item.strip() for item in args.datasets.split(",") if item.strip()}
    rows: list[dict[str, object]] = []

    for file_path in list_dataset_files(dataset_filter):
        bundle = read_dataset(file_path)

        rules = resolve_dataset_rules(bundle.dataset_name, config)
        for column in bundle.numeric_columns:
            series = pd.to_numeric(bundle.frame[column], errors="coerce")
            rows.extend(
                detect_series_artifacts(
                    dataset_name=bundle.dataset_name,
                    file_path=bundle.file_path,
                    variable=column,
                    series=series,
                    rules=rules,
                )
            )

    rows.sort(
        key=lambda row: (
            row["dataset_name"],
            row["variable"],
            row["start_idx"],
            row["artifact_type"],
        )
    )
    output_path = STATISTIC_RESULTS_DIR / "artifact_candidates.csv"
    write_csv(output_path, rows, fieldnames=artifact_fieldnames())
    print(f"Wrote {len(rows)} artifact candidates to {output_path}")


def detect_series_artifacts(
    dataset_name: str,
    file_path: Path,
    variable: str,
    series: pd.Series,
    rules: dict[str, Any],
) -> list[dict[str, object]]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, object]] = []

    for start_idx, end_idx, score in find_zero_blocks(values, rules):
        rows.append(make_row(dataset_name, file_path, variable, start_idx, end_idx, "zero_block", score))
    for start_idx, end_idx, score in find_flat_runs(values, rules):
        rows.append(make_row(dataset_name, file_path, variable, start_idx, end_idx, "flat_run", score))
    for start_idx, end_idx, score in find_near_constant_segments(values, rules):
        rows.append(
            make_row(
                dataset_name,
                file_path,
                variable,
                start_idx,
                end_idx,
                "near_constant_segment",
                score,
            )
        )
    for start_idx, end_idx, score in find_suspicious_repetition(values, rules):
        rows.append(
            make_row(
                dataset_name,
                file_path,
                variable,
                start_idx,
                end_idx,
                "suspicious_repetition",
                score,
            )
        )
    return rows


def make_row(
    dataset_name: str,
    file_path: Path,
    variable: str,
    start_idx: int,
    end_idx: int,
    artifact_type: str,
    score: float,
) -> dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "file_path": relative_path(file_path),
        "variable": variable,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "length": end_idx - start_idx + 1,
        "artifact_type": artifact_type,
        "score": round(float(score), 6),
    }


def find_zero_blocks(values: np.ndarray, rules: dict[str, Any]) -> list[tuple[int, int, float]]:
    valid = np.isfinite(values)
    zero_mask = valid & (np.abs(values) <= rules["zero_tolerance"])
    segments = mask_to_segments(zero_mask, rules["zero_block_min_length"])
    return [
        (start, end, min((end - start + 1) / rules["zero_block_min_length"], 10.0))
        for start, end in segments
    ]


def find_flat_runs(values: np.ndarray, rules: dict[str, Any]) -> list[tuple[int, int, float]]:
    valid = np.isfinite(values)
    if valid.sum() <= 1:
        return []
    same_as_previous = np.zeros_like(values, dtype=bool)
    diffs = np.abs(np.diff(values))
    same_as_previous[1:] = valid[1:] & valid[:-1] & (diffs <= rules["flat_tolerance"])
    segments = mask_to_segments(same_as_previous, rules["flat_run_min_length"])

    flat_segments: list[tuple[int, int, float]] = []
    for start, end in segments:
        segment_values = values[start : end + 1]
        if np.all(np.isfinite(segment_values)) and np.all(np.abs(segment_values) <= rules["zero_tolerance"]):
            continue
        score = min((end - start + 1) / rules["flat_run_min_length"], 10.0)
        flat_segments.append((start, end, score))
    return flat_segments


def find_near_constant_segments(values: np.ndarray, rules: dict[str, Any]) -> list[tuple[int, int, float]]:
    window = int(rules["near_constant_window"])
    min_length = int(rules["near_constant_min_length"])
    finite_values = values[np.isfinite(values)]
    if window < 2 or len(values) < window or len(finite_values) < window:
        return []

    rolling_std = pd.Series(values).rolling(window=window, min_periods=window).std(ddof=0)
    scale = max(float(np.nanstd(finite_values)), rules["near_constant_min_std"])
    threshold = max(scale * rules["near_constant_std_ratio"], rules["near_constant_min_std"])
    mask = rolling_std.fillna(np.inf).to_numpy() <= threshold
    segments = mask_to_segments(mask, min_length)

    rows: list[tuple[int, int, float]] = []
    for start, end in segments:
        segment_std = float(np.nanstd(values[start : end + 1]))
        score = max(0.0, 1.0 - segment_std / threshold) + (end - start + 1) / max(min_length, 1)
        rows.append((start, end, score))
    return rows


def find_suspicious_repetition(values: np.ndarray, rules: dict[str, Any]) -> list[tuple[int, int, float]]:
    window = int(rules["repetition_window"])
    min_length = int(rules["repetition_min_length"])
    finite_values = values[np.isfinite(values)]
    if window < 2 or len(values) < 2 * window or len(finite_values) < 2 * window:
        return []

    scale = max(float(np.nanstd(finite_values)), rules["repetition_min_abs_tolerance"])
    tolerance = max(scale * rules["repetition_tolerance_ratio"], rules["repetition_min_abs_tolerance"])
    comparable = np.isfinite(values[:-window]) & np.isfinite(values[window:])
    repeated = comparable & (np.abs(values[:-window] - values[window:]) <= tolerance)
    segments = mask_to_segments(repeated, window)

    rows: list[tuple[int, int, float]] = []
    for start, repeated_end in segments:
        end = repeated_end + window
        if end - start + 1 < min_length:
            continue
        lhs = values[start : repeated_end + 1]
        rhs = values[start + window : end + 1]
        mean_gap = float(np.nanmean(np.abs(lhs - rhs)))
        score = max(0.0, 1.0 - mean_gap / tolerance) + (end - start + 1) / max(min_length, 1)
        rows.append((start, end, score))
    return rows


def mask_to_segments(mask: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            end = index - 1
            if end - start + 1 >= min_length:
                segments.append((start, end))
            start = None
    if start is not None:
        end = len(mask) - 1
        if end - start + 1 >= min_length:
            segments.append((start, end))
    return segments


def artifact_fieldnames() -> list[str]:
    return [
        "dataset_name",
        "file_path",
        "variable",
        "start_idx",
        "end_idx",
        "length",
        "artifact_type",
        "score",
    ]


if __name__ == "__main__":
    detect_artifacts()
