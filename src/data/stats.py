from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd


SOLAR_PHASE_STEPS_PER_DAY = 144
SOLAR_NIGHT_ZERO_THRESHOLD = 0.95
SOLAR_ACTIVE_ZERO_THRESHOLD = 0.05
FloatArray = npt.NDArray[np.float64]


def dataset_missing_ratio(frame: pd.DataFrame, numeric_columns: list[str]) -> float:
    if not numeric_columns:
        return 0.0
    missing_matrix = frame.loc[:, numeric_columns].isna().to_numpy(dtype=np.float64)
    return float(np.mean(missing_matrix))


def row_zero_ratio(frame: pd.DataFrame, numeric_columns: list[str], tolerance: float = 1e-12) -> pd.Series:
    if not numeric_columns:
        return pd.Series(dtype=np.float64, name="row_zero_ratio")

    numeric_frame = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric_frame.to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(values)
    valid_counts = finite_mask.sum(axis=1)
    zero_counts = (finite_mask & (np.abs(values) <= tolerance)).sum(axis=1)
    ratios = np.divide(
        zero_counts,
        valid_counts,
        out=np.zeros_like(zero_counts, dtype=np.float64),
        where=valid_counts > 0,
    )
    return pd.Series(ratios, index=frame.index, name="row_zero_ratio")


def compute_solar_phase_profile(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    steps_per_day: int = SOLAR_PHASE_STEPS_PER_DAY,
) -> pd.DataFrame:
    row_zero = row_zero_ratio(frame, numeric_columns)
    phase_idx = np.arange(len(row_zero), dtype=int) % steps_per_day
    phase_frame = pd.DataFrame({"phase_idx": phase_idx, "row_zero_ratio": row_zero.to_numpy(dtype=np.float64)})
    profile = (
        phase_frame.groupby("phase_idx", as_index=False)["row_zero_ratio"]
        .agg(["mean", "std", "min", "median", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "row_zero_ratio_mean",
                "std": "row_zero_ratio_std",
                "min": "row_zero_ratio_min",
                "median": "row_zero_ratio_median",
                "max": "row_zero_ratio_max",
                "count": "n_rows",
            }
        )
    )

    phase_groups: list[str] = []
    artifact_groups: list[str] = []
    confidences: list[float] = []
    for mean_value in profile["row_zero_ratio_mean"].tolist():
        phase_group, artifact_group, confidence = classify_solar_phase(float(mean_value))
        phase_groups.append(phase_group)
        artifact_groups.append(artifact_group)
        confidences.append(round(confidence, 6))

    profile["row_active_ratio_mean"] = 1.0 - profile["row_zero_ratio_mean"]
    profile["phase_group"] = phase_groups
    profile["artifact_group"] = artifact_groups
    profile["confidence"] = confidences
    return profile.loc[
        :,
        [
            "phase_idx",
            "n_rows",
            "row_zero_ratio_mean",
            "row_active_ratio_mean",
            "row_zero_ratio_std",
            "row_zero_ratio_min",
            "row_zero_ratio_median",
            "row_zero_ratio_max",
            "phase_group",
            "artifact_group",
            "confidence",
        ],
    ].copy()


def classify_solar_phase(row_zero_ratio_mean: float) -> tuple[str, str, float]:
    if row_zero_ratio_mean >= SOLAR_NIGHT_ZERO_THRESHOLD:
        confidence = min(
            1.0,
            max(
                0.0,
                (row_zero_ratio_mean - SOLAR_NIGHT_ZERO_THRESHOLD)
                / max(1.0 - SOLAR_NIGHT_ZERO_THRESHOLD, 1e-12),
            ),
        )
        return "night", "night_zero_band", confidence

    if row_zero_ratio_mean <= SOLAR_ACTIVE_ZERO_THRESHOLD:
        confidence = min(
            1.0,
            max(
                0.0,
                (SOLAR_ACTIVE_ZERO_THRESHOLD - row_zero_ratio_mean)
                / max(SOLAR_ACTIVE_ZERO_THRESHOLD, 1e-12),
            ),
        )
        return "active", "active_period", confidence

    distance_to_boundary = min(
        row_zero_ratio_mean - SOLAR_ACTIVE_ZERO_THRESHOLD,
        SOLAR_NIGHT_ZERO_THRESHOLD - row_zero_ratio_mean,
    )
    transition_span = max((SOLAR_NIGHT_ZERO_THRESHOLD - SOLAR_ACTIVE_ZERO_THRESHOLD) / 2.0, 1e-12)
    confidence = min(1.0, max(0.0, distance_to_boundary / transition_span))
    return "transition", "phase_transition_band", confidence


def build_solar_phase_annotations(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    steps_per_day: int = SOLAR_PHASE_STEPS_PER_DAY,
) -> pd.DataFrame:
    row_zero = row_zero_ratio(frame, numeric_columns)
    if row_zero.empty:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "start_idx",
                    "end_idx",
                    "length",
                    "phase_idx_start",
                    "phase_idx_end",
                    "phase_group",
                    "artifact_group",
                    "row_zero_ratio_mean",
                    "phase_zero_ratio_mean",
                    "confidence",
                ]
            )
        )

    profile = compute_solar_phase_profile(frame, numeric_columns, steps_per_day=steps_per_day)
    profile_lookup = profile.set_index("phase_idx")
    phase_idx = np.arange(len(row_zero), dtype=int) % steps_per_day

    per_row = pd.DataFrame(
        {
            "row_idx": np.arange(len(row_zero), dtype=int),
            "phase_idx": phase_idx,
            "row_zero_ratio": row_zero.to_numpy(dtype=np.float64),
        }
    )
    per_row["phase_group"] = per_row["phase_idx"].map(profile_lookup["phase_group"])
    per_row["artifact_group"] = per_row["phase_idx"].map(profile_lookup["artifact_group"])
    per_row["confidence"] = per_row["phase_idx"].map(profile_lookup["confidence"]).astype(float)
    per_row["phase_zero_ratio_mean"] = per_row["phase_idx"].map(profile_lookup["row_zero_ratio_mean"]).astype(float)

    rows: list[dict[str, Any]] = []
    start_idx = 0
    current_group = str(per_row.at[0, "phase_group"])
    current_artifact = str(per_row.at[0, "artifact_group"])

    for row_idx in range(1, len(per_row)):
        phase_group = str(per_row.at[row_idx, "phase_group"])
        artifact_group = str(per_row.at[row_idx, "artifact_group"])
        if phase_group == current_group and artifact_group == current_artifact:
            continue
        rows.append(build_phase_segment_row(per_row, start_idx, row_idx - 1))
        start_idx = row_idx
        current_group = phase_group
        current_artifact = artifact_group

    rows.append(build_phase_segment_row(per_row, start_idx, len(per_row) - 1))
    return pd.DataFrame(rows)


def build_phase_segment_row(per_row: pd.DataFrame, start_idx: int, end_idx: int) -> dict[str, Any]:
    segment = per_row.iloc[start_idx : end_idx + 1]
    return {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "length": end_idx - start_idx + 1,
        "phase_idx_start": int(segment["phase_idx"].iloc[0]),
        "phase_idx_end": int(segment["phase_idx"].iloc[-1]),
        "phase_group": str(segment["phase_group"].iloc[0]),
        "artifact_group": str(segment["artifact_group"].iloc[0]),
        "row_zero_ratio_mean": round(float(segment["row_zero_ratio"].mean()), 6),
        "phase_zero_ratio_mean": round(float(segment["phase_zero_ratio_mean"].iloc[0]), 6),
        "confidence": round(float(segment["confidence"].mean()), 6),
    }


def zero_ratio(series: pd.Series, tolerance: float = 1e-12) -> float:
    values = _series_to_float_array(series)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0
    return float(np.mean(np.abs(finite_values) <= tolerance))


def constant_ratio(series: pd.Series, tolerance: float = 1e-12) -> float:
    values = _series_to_float_array(series)
    finite_values = values[np.isfinite(values)]
    if finite_values.size <= 1:
        return 0.0
    diffs = np.abs(np.diff(finite_values))
    return float(np.mean(diffs <= tolerance))


def _series_to_float_array(series: pd.Series) -> FloatArray:
    numeric = pd.to_numeric(series, errors="coerce")
    return cast(FloatArray, np.asarray(numeric, dtype=np.float64))


def safe_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)
