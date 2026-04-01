from __future__ import annotations

import json

import numpy as np
import pandas as pd


VIEW_FLAG_COLUMNS = {
    "raw": "is_raw_view",
    "natural_clean": "is_raw_view",
    "anchor_clean": "is_anchor_clean_view",
    "conservative_clean": "is_conservative_clean_view",
    "clean_like": "is_conservative_clean_view",
    "intervened": "is_intervened_view",
    "flagged_group": "is_group_controlled_view",
    "balanced": "is_phase_balanced_view",
    "active_only": "is_active_only_view",
    "daytime_only": "is_daytime_only_view",
}


def resolve_dataset_splits(dataset_name: str, n_rows: int) -> dict[str, tuple[int, int]]:
    if dataset_name in {"ETTh1", "ETTh2"}:
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = min(n_rows, val_end + 4 * 30 * 24)
        return {
            "train": (0, train_end - 1),
            "val": (train_end, val_end - 1),
            "test": (val_end, test_end - 1),
        }
    if dataset_name in {"ETTm1", "ETTm2"}:
        factor = 4
        train_end = 12 * 30 * 24 * factor
        val_end = train_end + 4 * 30 * 24 * factor
        test_end = min(n_rows, val_end + 4 * 30 * 24 * factor)
        return {
            "train": (0, train_end - 1),
            "val": (train_end, val_end - 1),
            "test": (val_end, test_end - 1),
        }
    train_end = int(np.floor(n_rows * 0.70))
    val_end = int(np.floor(n_rows * 0.80))
    return {
        "train": (0, max(train_end - 1, 0)),
        "val": (train_end, max(val_end - 1, train_end)),
        "test": (val_end, max(n_rows - 1, val_end)),
    }


def build_window_id(dataset_name: str, lookback: int, horizon: int, split_name: str, target_start: int) -> str:
    return f"{dataset_name}|L{lookback}|H{horizon}|split={split_name}|t={target_start}"


def prefix_sum(values: np.ndarray) -> np.ndarray:
    return np.concatenate([np.array([0.0], dtype=float), np.cumsum(values.astype(float))])


def sum_between(prefix: np.ndarray, start_idx: int, end_idx: int) -> float:
    if end_idx < start_idx:
        return 0.0
    return float(prefix[end_idx + 1] - prefix[start_idx])


def unique_ordered(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def parse_artifact_ids(value: str | float | int | None) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def severity_bin(severity: float) -> str:
    if severity >= 0.80:
        return "high"
    if severity >= 0.50:
        return "medium"
    if severity > 0:
        return "low"
    return "none"


def n_variables_bin(n_variables: int) -> str:
    if n_variables >= 8:
        return "8+"
    if n_variables >= 4:
        return "4-7"
    if n_variables >= 2:
        return "2-3"
    return "1"


def phase_mix_bin(target_active: float, target_transition: float, target_night: float) -> str:
    if target_active >= 0.95:
        return "active_pure"
    if target_night == 0 and target_active + target_transition >= 0.80:
        return "daylike_mixed"
    if target_night >= 0.80:
        return "night_heavy"
    return "phase_mixed"


def deterministic_subsample(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.sort_values("target_start").reset_index(drop=True).copy()
    picked = np.linspace(0, len(df) - 1, num=max_rows, dtype=int)
    picked = np.unique(picked)
    return df.iloc[picked].sort_values("target_start").reset_index(drop=True).copy()


def select_view_rows(view_df: pd.DataFrame, split_name: str, view_name: str, max_rows: int | None = None) -> pd.DataFrame:
    flag_col = VIEW_FLAG_COLUMNS[view_name]
    subset = view_df.loc[(view_df["split_name"] == split_name) & (view_df[flag_col] == 1)].copy()
    return deterministic_subsample(subset, max_rows=max_rows)


def resolve_validation_rows(view_df: pd.DataFrame, train_view_name: str, max_val_rows: int | None) -> tuple[str, pd.DataFrame]:
    preferred = select_view_rows(view_df, split_name="val", view_name=train_view_name, max_rows=max_val_rows)
    if not preferred.empty:
        return train_view_name, preferred

    fallback_raw = select_view_rows(view_df, split_name="val", view_name="raw", max_rows=max_val_rows)
    if not fallback_raw.empty:
        return "raw", fallback_raw

    train_rows = select_view_rows(view_df, split_name="train", view_name=train_view_name, max_rows=max_val_rows)
    if train_rows.empty:
        return train_view_name, train_rows

    holdout = train_rows.iloc[:: max(1, len(train_rows) // max(1, min(512, len(train_rows))))].copy()
    return f"{train_view_name}_train_holdout", holdout.reset_index(drop=True)
