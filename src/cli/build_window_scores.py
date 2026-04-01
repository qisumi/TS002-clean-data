from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data import STATISTIC_RESULTS_DIR, ensure_project_directories
from utils import progress
from views import (
    build_window_id,
    load_view_spec,
    n_variables_bin,
    phase_mix_bin,
    prefix_sum,
    resolve_dataset_splits,
    severity_bin,
    sum_between,
    unique_ordered,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build window-level contamination scores from final artifact events.")
    parser.add_argument("--registry", default=str(STATISTIC_RESULTS_DIR / "dataset_registry.csv"))
    parser.add_argument("--events", default=str(STATISTIC_RESULTS_DIR / "final_artifact_events.csv"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ETTh2", "ETTm2", "solar_AL", "ETTh1", "ETTm1"],
    )
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192])
    parser.add_argument("--out-dir", default=str(STATISTIC_RESULTS_DIR / "window_scores"))
    parser.add_argument("--spec", default=str(Path("configs") / "view_specs.yaml"))
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def event_weight(row: pd.Series, weight_cfg: dict[str, float]) -> float:
    key = f"{row['validity']}_{row['exploitability']}"
    base_weight = float(weight_cfg.get(key, 0.0))
    return float(base_weight * float(row["confidence"]) * max(0.25, float(row["severity"])))


def dominant_phase_from_shares(active: float, transition: float, night: float) -> str:
    shares = {"active": active, "transition": transition, "night": night}
    phase_name = max(shares, key=shares.get)
    if shares[phase_name] <= 0:
        return "NA"
    return phase_name


def prepare_accumulator(target_starts: np.ndarray) -> dict[str, Any]:
    length = len(target_starts)
    return {
        "target_starts": target_starts,
        "input_event_indices": [[] for _ in range(length)],
        "target_event_indices": [[] for _ in range(length)],
        "max_event_weight_input": np.zeros(length, dtype=float),
        "max_event_weight_target": np.zeros(length, dtype=float),
        "has_corrupted_input": np.zeros(length, dtype=int),
        "has_corrupted_target": np.zeros(length, dtype=int),
        "has_suspicious_input": np.zeros(length, dtype=int),
        "has_suspicious_target": np.zeros(length, dtype=int),
        "has_valid_high_input": np.zeros(length, dtype=int),
        "has_valid_high_target": np.zeros(length, dtype=int),
        "has_multivar_severe_input": np.zeros(length, dtype=int),
        "has_multivar_severe_target": np.zeros(length, dtype=int),
        "has_ot_severe_target": np.zeros(length, dtype=int),
        "repairable_input_overlap": np.zeros(length, dtype=int),
        "unrecoverable_input_overlap": np.zeros(length, dtype=int),
        "has_unrecoverable_target": np.zeros(length, dtype=int),
    }


def update_window_slice(values: np.ndarray, start_idx: int, end_idx: int, flag: int) -> None:
    if start_idx > end_idx:
        return
    values[start_idx : end_idx + 1] = np.maximum(values[start_idx : end_idx + 1], flag)


def update_max_slice(values: np.ndarray, start_idx: int, end_idx: int, weight: float) -> None:
    if start_idx > end_idx:
        return
    values[start_idx : end_idx + 1] = np.maximum(values[start_idx : end_idx + 1], weight)


def append_event_indices(target_lists: list[list[int]], start_idx: int, end_idx: int, event_idx: int) -> None:
    for pos in range(start_idx, end_idx + 1):
        target_lists[pos].append(event_idx)


def accumulate_window_membership(
    dataset_events: pd.DataFrame,
    lookback: int,
    horizon: int,
    split_accumulators: dict[str, dict[str, Any]],
    weight_cfg: dict[str, float],
    show_progress: bool = False,
    progress_desc: str = "",
) -> pd.DataFrame:
    event_df = dataset_events.reset_index(drop=True).copy()
    event_df["event_weight"] = event_df.apply(lambda row: event_weight(row, weight_cfg=weight_cfg), axis=1)

    iterator = progress(
        list(event_df.iterrows()),
        total=len(event_df),
        desc=progress_desc or "005/window event-map",
        disable=not show_progress,
    )
    for event_idx, row in iterator:
        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])
        evt_weight = float(row["event_weight"])
        is_corrupted = int(row["validity"] == "corrupted")
        is_suspicious = int(row["validity"] == "suspicious")
        is_valid_high = int(row["validity"] == "valid" and row["exploitability"] == "high")
        is_multivar_severe = int(int(row["n_variables"]) >= 2 and float(row["severity"]) >= 0.7)
        has_ot_severe = int("OT" in str(row["variables"]).split(",") and float(row["severity"]) >= 0.7)
        is_repairable = int(row["recoverability"] == "repairable")
        is_unrecoverable = int(row["validity"] != "valid" and row["recoverability"] in {"mask_only", "unrecoverable"})

        for accumulator in split_accumulators.values():
            target_starts = accumulator["target_starts"]
            if len(target_starts) == 0:
                continue
            t_min = int(target_starts[0])
            t_max = int(target_starts[-1])

            input_lo = max(t_min, start_idx + 1)
            input_hi = min(t_max, end_idx + lookback)
            if input_lo <= input_hi:
                start_pos = input_lo - t_min
                end_pos = input_hi - t_min
                append_event_indices(accumulator["input_event_indices"], start_pos, end_pos, event_idx)
                update_max_slice(accumulator["max_event_weight_input"], start_pos, end_pos, evt_weight)
                update_window_slice(accumulator["has_corrupted_input"], start_pos, end_pos, is_corrupted)
                update_window_slice(accumulator["has_suspicious_input"], start_pos, end_pos, is_suspicious)
                update_window_slice(accumulator["has_valid_high_input"], start_pos, end_pos, is_valid_high)
                update_window_slice(accumulator["has_multivar_severe_input"], start_pos, end_pos, is_multivar_severe)
                update_window_slice(accumulator["repairable_input_overlap"], start_pos, end_pos, is_repairable)
                update_window_slice(accumulator["unrecoverable_input_overlap"], start_pos, end_pos, is_unrecoverable)

            target_lo = max(t_min, start_idx - horizon + 1)
            target_hi = min(t_max, end_idx)
            if target_lo <= target_hi:
                start_pos = target_lo - t_min
                end_pos = target_hi - t_min
                append_event_indices(accumulator["target_event_indices"], start_pos, end_pos, event_idx)
                update_max_slice(accumulator["max_event_weight_target"], start_pos, end_pos, evt_weight)
                update_window_slice(accumulator["has_corrupted_target"], start_pos, end_pos, is_corrupted)
                update_window_slice(accumulator["has_suspicious_target"], start_pos, end_pos, is_suspicious)
                update_window_slice(accumulator["has_valid_high_target"], start_pos, end_pos, is_valid_high)
                update_window_slice(accumulator["has_multivar_severe_target"], start_pos, end_pos, is_multivar_severe)
                update_window_slice(accumulator["has_ot_severe_target"], start_pos, end_pos, has_ot_severe)
                update_window_slice(accumulator["has_unrecoverable_target"], start_pos, end_pos, is_unrecoverable)

    return event_df


def build_row_level_arrays(dataset_name: str, dataset_events: pd.DataFrame, n_rows: int, weight_cfg: dict[str, float]) -> dict[str, np.ndarray]:
    weight_diff = np.zeros(n_rows + 1, dtype=float)
    phase_diffs = {
        "active": np.zeros(n_rows + 1, dtype=float),
        "transition": np.zeros(n_rows + 1, dtype=float),
        "night": np.zeros(n_rows + 1, dtype=float),
    }

    for row in dataset_events.itertuples(index=False):
        weight = event_weight(pd.Series(row._asdict()), weight_cfg)
        start_idx = int(row.start_idx)
        end_idx = int(row.end_idx)
        weight_diff[start_idx] += weight
        if end_idx + 1 < len(weight_diff):
            weight_diff[end_idx + 1] -= weight

        if dataset_name == "solar_AL" and str(row.source_kind) == "phase_annotation":
            phase_group = str(row.phase_group)
            if phase_group in phase_diffs:
                phase_diffs[phase_group][start_idx] += 1.0
                if end_idx + 1 < len(phase_diffs[phase_group]):
                    phase_diffs[phase_group][end_idx + 1] -= 1.0

    arrays = {
        "row_weight": np.cumsum(weight_diff[:-1]),
        "row_phase_active": np.cumsum(phase_diffs["active"][:-1]),
        "row_phase_transition": np.cumsum(phase_diffs["transition"][:-1]),
        "row_phase_night": np.cumsum(phase_diffs["night"][:-1]),
    }
    return arrays


def build_window_rows(
    dataset_name: str,
    dataset_events: pd.DataFrame,
    n_rows: int,
    lookback: int,
    horizon: int,
    weight_cfg: dict[str, float],
    show_progress: bool = False,
) -> pd.DataFrame:
    row_arrays = build_row_level_arrays(dataset_name, dataset_events, n_rows, weight_cfg)
    row_weight_prefix = prefix_sum(row_arrays["row_weight"])
    phase_prefix = {
        "active": prefix_sum(row_arrays["row_phase_active"]),
        "transition": prefix_sum(row_arrays["row_phase_transition"]),
        "night": prefix_sum(row_arrays["row_phase_night"]),
    }

    split_accumulators: dict[str, dict[str, Any]] = {}
    for split_name, (split_start, split_end) in resolve_dataset_splits(dataset_name, n_rows).items():
        target_starts = np.arange(split_start + lookback, split_end - horizon + 2, dtype=int)
        split_accumulators[split_name] = prepare_accumulator(target_starts)

    event_df = accumulate_window_membership(
        dataset_events,
        lookback,
        horizon,
        split_accumulators,
        weight_cfg,
        show_progress=show_progress,
        progress_desc=f"005/{dataset_name} H{horizon} event-map",
    )
    artifact_ids = event_df["artifact_id"].tolist()
    artifact_groups = event_df["artifact_group"].tolist()
    severities = event_df["severity"].astype(float).tolist()
    n_variables_list = event_df["n_variables"].astype(int).tolist()

    rows: list[dict[str, Any]] = []
    for split_name, accumulator in split_accumulators.items():
        for pos, target_start in enumerate(accumulator["target_starts"]):
            input_start = int(target_start - lookback)
            input_end = int(target_start - 1)
            target_end = int(target_start + horizon - 1)
            input_contam = np.clip(sum_between(row_weight_prefix, input_start, input_end) / max(lookback, 1), 0.0, 1.5)
            target_contam = np.clip(sum_between(row_weight_prefix, int(target_start), target_end) / max(horizon, 1), 0.0, 1.5)

            phase_share_input_active = sum_between(phase_prefix["active"], input_start, input_end) / max(lookback, 1)
            phase_share_input_transition = sum_between(phase_prefix["transition"], input_start, input_end) / max(lookback, 1)
            phase_share_input_night = sum_between(phase_prefix["night"], input_start, input_end) / max(lookback, 1)
            phase_share_target_active = sum_between(phase_prefix["active"], int(target_start), target_end) / max(horizon, 1)
            phase_share_target_transition = sum_between(phase_prefix["transition"], int(target_start), target_end) / max(horizon, 1)
            phase_share_target_night = sum_between(phase_prefix["night"], int(target_start), target_end) / max(horizon, 1)

            input_event_indices = unique_ordered(accumulator["input_event_indices"][pos])
            target_event_indices = unique_ordered(accumulator["target_event_indices"][pos])
            all_event_indices = unique_ordered(input_event_indices + target_event_indices)
            focus_indices = target_event_indices if target_event_indices else input_event_indices
            major_idx = max(
                focus_indices,
                key=lambda idx: (severities[idx], float(event_df.at[idx, "event_weight"]), int(event_df.at[idx, "length"])),
            ) if focus_indices else None

            artifact_group_major = artifact_groups[major_idx] if major_idx is not None else "NA"
            major_severity = severities[major_idx] if major_idx is not None else 0.0
            max_n_variables = max((n_variables_list[idx] for idx in focus_indices), default=0)

            rows.append(
                {
                    "dataset_name": dataset_name,
                    "split_name": split_name,
                    "source_split": split_name,
                    "lookback": lookback,
                    "horizon": horizon,
                    "window_id": build_window_id(dataset_name, lookback, horizon, split_name, int(target_start)),
                    "input_start": input_start,
                    "input_end": input_end,
                    "target_start": int(target_start),
                    "target_end": target_end,
                    "input_contam_score": round(float(input_contam), 6),
                    "target_contam_score": round(float(target_contam), 6),
                    "n_events_input": len(input_event_indices),
                    "n_events_target": len(target_event_indices),
                    "max_event_weight_input": round(float(accumulator["max_event_weight_input"][pos]), 6),
                    "max_event_weight_target": round(float(accumulator["max_event_weight_target"][pos]), 6),
                    "has_corrupted_input": int(accumulator["has_corrupted_input"][pos]),
                    "has_corrupted_target": int(accumulator["has_corrupted_target"][pos]),
                    "has_suspicious_input": int(accumulator["has_suspicious_input"][pos]),
                    "has_suspicious_target": int(accumulator["has_suspicious_target"][pos]),
                    "has_valid_high_input": int(accumulator["has_valid_high_input"][pos]),
                    "has_valid_high_target": int(accumulator["has_valid_high_target"][pos]),
                    "has_multivar_severe_input": int(accumulator["has_multivar_severe_input"][pos]),
                    "has_multivar_severe_target": int(accumulator["has_multivar_severe_target"][pos]),
                    "has_ot_severe_target": int(accumulator["has_ot_severe_target"][pos]),
                    "repairable_input_overlap": int(accumulator["repairable_input_overlap"][pos]),
                    "unrecoverable_input_overlap": int(accumulator["unrecoverable_input_overlap"][pos]),
                    "has_unrecoverable_target": int(accumulator["has_unrecoverable_target"][pos]),
                    "artifact_ids_input": ",".join(artifact_ids[idx] for idx in input_event_indices),
                    "artifact_ids_target": ",".join(artifact_ids[idx] for idx in target_event_indices),
                    "artifact_ids": ",".join(artifact_ids[idx] for idx in all_event_indices),
                    "artifact_group_major": artifact_group_major,
                    "severity_bin": severity_bin(float(major_severity)),
                    "n_variables_bin": n_variables_bin(int(max_n_variables)),
                    "phase_share_input_active": round(float(phase_share_input_active), 6),
                    "phase_share_input_transition": round(float(phase_share_input_transition), 6),
                    "phase_share_input_night": round(float(phase_share_input_night), 6),
                    "phase_share_target_active": round(float(phase_share_target_active), 6),
                    "phase_share_target_transition": round(float(phase_share_target_transition), 6),
                    "phase_share_target_night": round(float(phase_share_target_night), 6),
                    "dominant_phase_input": dominant_phase_from_shares(
                        float(phase_share_input_active),
                        float(phase_share_input_transition),
                        float(phase_share_input_night),
                    ),
                    "dominant_phase_target": dominant_phase_from_shares(
                        float(phase_share_target_active),
                        float(phase_share_target_transition),
                        float(phase_share_target_night),
                    ),
                    "phase_mix_bin_target": phase_mix_bin(
                        float(phase_share_target_active),
                        float(phase_share_target_transition),
                        float(phase_share_target_night),
                    ),
                    "has_active_suspicious_target": int(
                        any(artifact_groups[idx].startswith("active_suspicious") for idx in target_event_indices)
                    ),
                    "is_flagged": int(bool(target_event_indices or input_event_indices)),
                    "notes": "",
                }
            )

    return pd.DataFrame(rows)


def run_build_window_scores(
    registry_path: Path,
    events_path: Path,
    datasets: list[str],
    lookback: int,
    horizons: list[int],
    out_dir: Path,
    spec_path: Path,
    show_progress: bool = True,
) -> list[Path]:
    ensure_project_directories()
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(registry_path)
    events = pd.read_csv(events_path)
    spec = load_view_spec(spec_path)
    weight_cfg = spec["defaults"]["weights"]

    written_paths: list[Path] = []
    work_items = [(dataset_name, int(horizon)) for dataset_name in datasets for horizon in horizons]
    for dataset_name, horizon in progress(
        work_items,
        total=len(work_items),
        desc="005/window_scores",
        disable=not show_progress,
    ):
        dataset_events = events[events["dataset_name"] == dataset_name].copy()
        if dataset_events.empty:
            continue
        registry_row = registry.loc[registry["dataset_name"] == dataset_name].iloc[0]
        n_rows = int(registry_row["n_rows"])
        window_df = build_window_rows(
            dataset_name=dataset_name,
            dataset_events=dataset_events,
            n_rows=n_rows,
            lookback=lookback,
            horizon=int(horizon),
            weight_cfg=weight_cfg,
            show_progress=show_progress,
        )
        out_path = out_dir / f"{dataset_name}_L{lookback}_H{horizon}.csv"
        window_df.to_csv(out_path, index=False)
        written_paths.append(out_path)
    return written_paths


def main() -> None:
    args = parse_args()
    run_build_window_scores(
        registry_path=Path(args.registry),
        events_path=Path(args.events),
        datasets=list(args.datasets),
        lookback=int(args.lookback),
        horizons=[int(item) for item in args.horizons],
        out_dir=Path(args.out_dir),
        spec_path=Path(args.spec),
        show_progress=not bool(args.no_progress),
    )


if __name__ == "__main__":
    main()
