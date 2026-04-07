from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.aif_shared import compute_aif_arg_table, compute_aif_ri_table


SETTING_KEYS = [
    "dataset_name",
    "backbone",
    "lookback",
    "horizon",
    "train_view_name",
    "eval_view_name",
    "seed",
    "checkpoint_variant",
]

WGR_KEYS = [
    "dataset_name",
    "backbone",
    "lookback",
    "horizon",
    "train_view_name",
    "eval_view_name",
]

MAINBOARD_DATASETS = {"ETTh1", "ETTm1", "exchange_rate", "weather", "solar_AL"}
BOUNDARY_DATASETS = {"ETTh2"}
APPENDIX_DATASETS = {"ETTm2", "electricity"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute AIF-Plus tables from shard window errors.")
    parser.add_argument("--window-errors", nargs="+", required=True)
    parser.add_argument("--results-online", nargs="+", required=True)
    parser.add_argument("--chunksize", type=int, default=250000)
    parser.add_argument("--merged-results-out", default=str(Path("results") / "aif_plus_v5_results_online.csv"))
    parser.add_argument("--merged-window-errors-out", default=str(Path("results") / "aif_plus_v5_window_errors.csv"))
    parser.add_argument("--results-out", default=str(Path("results") / "aif_plus_v5_results.csv"))
    parser.add_argument("--diagnostics-out", default=str(Path("results") / "aif_plus_v5_validity_diagnostics.csv"))
    parser.add_argument("--arg-out", default=str(Path("results") / "aif_plus_v5_artifact_reliance_gap.csv"))
    parser.add_argument("--wgr-out", default=str(Path("results") / "aif_plus_v5_worst_group_risk.csv"))
    parser.add_argument("--ri-out", default=str(Path("results") / "aif_plus_v5_ranking_instability.csv"))
    parser.add_argument("--mainboard-out", default=str(Path("results") / "aif_plus_v5_mainboard.csv"))
    parser.add_argument("--boundary-board-out", default=str(Path("results") / "aif_plus_v5_boundary_board.csv"))
    parser.add_argument("--appendix-board-out", default=str(Path("results") / "aif_plus_v5_appendix_board.csv"))
    parser.add_argument("--summary-out", default=str(Path("reports") / "aif_plus_v5_summary.md"))
    parser.add_argument("--summary-title", default="AIF-Plus Summary")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_key_value(column: str, value: Any) -> Any:
    if column in {"lookback", "horizon", "seed"}:
        if pd.isna(value):
            return 0
        return int(value)
    if pd.isna(value):
        return ""
    return str(value)


def row_to_setting_key(row: pd.Series | dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    return tuple(normalize_key_value(column, row[column]) for column in columns)


def empty_wgr_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dataset_name",
            "backbone",
            "lookback",
            "horizon",
            "train_view_name",
            "eval_view_name",
            "worst_group",
            "mean_error",
            "WGR",
            "WGR_gap",
            "n_groups",
            "n_eval_windows",
        ]
    )


def sort_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return frame.reset_index(drop=True)
    return frame.sort_values(existing).reset_index(drop=True)


def coerce_validity(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    lowered = series.fillna("").astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y", "on"})


def read_results_online(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = sort_frame(merged, SETTING_KEYS)
    if set(SETTING_KEYS).issubset(merged.columns):
        merged = merged.drop_duplicates(subset=SETTING_KEYS, keep="last").reset_index(drop=True)
    return merged


def resolve_board(row: dict[str, Any]) -> tuple[str, str]:
    dataset_name = str(row.get("dataset_name", ""))
    n_valid_windows = int(row.get("n_valid_windows", 0) or 0)
    if n_valid_windows <= 0:
        return "appendix", "n_valid_windows=0"
    if dataset_name in BOUNDARY_DATASETS:
        return "boundary", "dataset_boundary"
    if dataset_name in APPENDIX_DATASETS:
        return "appendix", "dataset_appendix"
    if dataset_name in MAINBOARD_DATASETS:
        return "mainboard", "dataset_mainboard"
    return "appendix", "dataset_unassigned"


def aggregate_window_errors(
    *,
    window_error_paths: list[Path],
    merged_window_errors_out: Path | None,
    chunksize: int,
) -> tuple[dict[tuple[Any, ...], dict[str, float]], dict[tuple[Any, ...], dict[str, float]], dict[tuple[Any, ...], dict[str, dict[str, float]]]]:
    diagnostics: dict[tuple[Any, ...], dict[str, float]] = {}
    setting_metrics: dict[tuple[Any, ...], dict[str, float]] = {}
    group_metrics: dict[tuple[Any, ...], dict[str, dict[str, float]]] = {}
    merged_written = False

    if merged_window_errors_out is not None and merged_window_errors_out.exists():
        merged_window_errors_out.unlink()

    for path in window_error_paths:
        if not path.exists():
            continue
        for chunk in pd.read_csv(path, low_memory=False, chunksize=max(int(chunksize), 1)):
            if merged_window_errors_out is not None:
                chunk.to_csv(merged_window_errors_out, mode="a", header=not merged_written, index=False)
                merged_written = True

            for metric_col in ["mae", "mse", "smape"]:
                if metric_col not in chunk.columns:
                    chunk[metric_col] = np.nan
            for key in SETTING_KEYS:
                if key not in chunk.columns:
                    chunk[key] = "" if key not in {"lookback", "horizon", "seed"} else 0
            if "group_key" not in chunk.columns:
                chunk["group_key"] = "NA"
            if "is_valid_metric" in chunk.columns:
                valid_flag = coerce_validity(chunk["is_valid_metric"])
            else:
                valid_flag = pd.Series(True, index=chunk.index)
            finite_flag = np.isfinite(chunk["mae"]) & np.isfinite(chunk["mse"]) & np.isfinite(chunk["smape"])
            valid_mask = valid_flag & finite_flag

            total_counts = chunk.groupby(SETTING_KEYS, dropna=False).size().reset_index(name="n_total_windows")
            for row in total_counts.itertuples(index=False):
                key = tuple(normalize_key_value(column, getattr(row, column)) for column in SETTING_KEYS)
                stats = diagnostics.setdefault(key, {"n_total_windows": 0, "n_valid_windows": 0})
                stats["n_total_windows"] += int(row.n_total_windows)

            valid_chunk = chunk.loc[valid_mask, SETTING_KEYS + ["group_key", "mae", "mse", "smape"]].copy()
            if valid_chunk.empty:
                continue

            valid_counts = valid_chunk.groupby(SETTING_KEYS, dropna=False).agg(
                n_valid_windows=("mae", "size"),
                mae_sum=("mae", "sum"),
                mse_sum=("mse", "sum"),
                smape_sum=("smape", "sum"),
            ).reset_index()
            for row in valid_counts.itertuples(index=False):
                key = tuple(normalize_key_value(column, getattr(row, column)) for column in SETTING_KEYS)
                diag = diagnostics.setdefault(key, {"n_total_windows": 0, "n_valid_windows": 0})
                diag["n_valid_windows"] += int(row.n_valid_windows)
                metrics = setting_metrics.setdefault(key, {"mae_sum": 0.0, "mse_sum": 0.0, "smape_sum": 0.0, "count": 0})
                metrics["mae_sum"] += float(row.mae_sum)
                metrics["mse_sum"] += float(row.mse_sum)
                metrics["smape_sum"] += float(row.smape_sum)
                metrics["count"] += int(row.n_valid_windows)

            group_counts = valid_chunk.groupby(SETTING_KEYS + ["group_key"], dropna=False).agg(
                mae_sum=("mae", "sum"),
                count=("mae", "size"),
            ).reset_index()
            for row in group_counts.itertuples(index=False):
                setting_key = tuple(normalize_key_value(column, getattr(row, column)) for column in SETTING_KEYS)
                group_name = normalize_key_value("group_key", getattr(row, "group_key"))
                group_state = group_metrics.setdefault(setting_key, {}).setdefault(
                    group_name,
                    {"mae_sum": 0.0, "count": 0},
                )
                group_state["mae_sum"] += float(row.mae_sum)
                group_state["count"] += int(row.count)

    return diagnostics, setting_metrics, group_metrics


def build_diagnostics_frame(
    all_keys: list[tuple[Any, ...]],
    diagnostics: dict[tuple[Any, ...], dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key in all_keys:
        stats = diagnostics.get(key, {})
        n_total_windows = int(stats.get("n_total_windows", 0))
        n_valid_windows = int(stats.get("n_valid_windows", 0))
        n_invalid_windows = max(n_total_windows - n_valid_windows, 0)
        invalid_ratio = float(n_invalid_windows) / float(n_total_windows) if n_total_windows > 0 else float("nan")
        row = {column: key[idx] for idx, column in enumerate(SETTING_KEYS)}
        row.update(
            {
                "n_total_windows": n_total_windows,
                "n_valid_windows": n_valid_windows,
                "n_invalid_windows": n_invalid_windows,
                "invalid_ratio": invalid_ratio,
            }
        )
        rows.append(row)
    return sort_frame(
        pd.DataFrame(
            rows,
            columns=SETTING_KEYS + ["n_total_windows", "n_valid_windows", "n_invalid_windows", "invalid_ratio"],
        ),
        SETTING_KEYS,
    )


def build_recomputed_results(
    results_online_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    setting_metrics: dict[tuple[Any, ...], dict[str, float]],
) -> pd.DataFrame:
    online_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    if not results_online_df.empty and set(SETTING_KEYS).issubset(results_online_df.columns):
        for record in results_online_df.to_dict(orient="records"):
            online_by_key[row_to_setting_key(record, SETTING_KEYS)] = dict(record)

    rows: list[dict[str, Any]] = []
    for diag in diagnostics_df.to_dict(orient="records"):
        key = row_to_setting_key(diag, SETTING_KEYS)
        base = dict(online_by_key.get(key, {column: diag[column] for column in SETTING_KEYS}))
        metrics = setting_metrics.get(key)
        if metrics is None or int(metrics.get("count", 0)) <= 0:
            mae = float("nan")
            mse = float("nan")
            smape = float("nan")
        else:
            count = int(metrics["count"])
            mae = float(metrics["mae_sum"] / count)
            mse = float(metrics["mse_sum"] / count)
            smape = float(metrics["smape_sum"] / count)

        base.update(diag)
        base["n_eval_windows"] = int(diag["n_total_windows"])
        base["mae"] = mae
        base["mse"] = mse
        base["smape"] = smape
        rows.append(base)

    result = pd.DataFrame(rows)
    ordered_prefix = [
        "dataset_name",
        "backbone",
        "lookback",
        "horizon",
        "train_view_name",
        "eval_view_name",
        "seed",
        "checkpoint_variant",
    ]
    ordered_suffix = [
        "n_train_windows",
        "n_eval_windows",
        "n_total_windows",
        "n_valid_windows",
        "n_invalid_windows",
        "invalid_ratio",
        "epochs_ran",
        "best_epoch",
        "best_val_mae",
        "best_val_mse",
        "best_val_score",
        "mae",
        "mse",
        "smape",
    ]
    ordered = [column for column in ordered_prefix + ordered_suffix if column in result.columns]
    remaining = [column for column in result.columns if column not in ordered]
    result = result[ordered + remaining]
    return sort_frame(result, SETTING_KEYS)


def build_wgr_frame(
    diagnostics_df: pd.DataFrame,
    setting_metrics: dict[tuple[Any, ...], dict[str, float]],
    group_metrics: dict[tuple[Any, ...], dict[str, dict[str, float]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for diag in diagnostics_df.to_dict(orient="records"):
        key = row_to_setting_key(diag, SETTING_KEYS)
        metrics = setting_metrics.get(key)
        if metrics is None or int(metrics.get("count", 0)) <= 0:
            continue
        groups = group_metrics.get(key, {})
        if not groups:
            continue
        mean_error = float(metrics["mae_sum"] / metrics["count"])
        worst_group = "NA"
        worst_value = float("-inf")
        for group_name, group_state in groups.items():
            if int(group_state["count"]) <= 0:
                continue
            group_mean = float(group_state["mae_sum"] / group_state["count"])
            if group_mean > worst_value:
                worst_value = group_mean
                worst_group = str(group_name)
        if not math.isfinite(worst_value):
            continue
        rows.append(
            {
                "dataset_name": diag["dataset_name"],
                "backbone": diag["backbone"],
                "lookback": int(diag["lookback"]),
                "horizon": int(diag["horizon"]),
                "train_view_name": diag["train_view_name"],
                "eval_view_name": diag["eval_view_name"],
                "worst_group": worst_group,
                "mean_error": mean_error,
                "WGR": worst_value,
                "WGR_gap": worst_value - mean_error,
                "n_groups": int(len(groups)),
                "n_eval_windows": int(metrics["count"]),
            }
        )
    if not rows:
        return empty_wgr_frame()
    return sort_frame(pd.DataFrame(rows), WGR_KEYS)


def build_board_frames(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if results_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    board_rows: list[dict[str, Any]] = []
    for row in results_df.to_dict(orient="records"):
        board_name, board_reason = resolve_board(row)
        board_rows.append({**row, "board_name": board_name, "board_reason": board_reason})
    board_df = pd.DataFrame(board_rows)
    mainboard_df = sort_frame(board_df.loc[board_df["board_name"] == "mainboard"].copy(), SETTING_KEYS)
    boundary_df = sort_frame(board_df.loc[board_df["board_name"] == "boundary"].copy(), SETTING_KEYS)
    appendix_df = sort_frame(board_df.loc[board_df["board_name"] == "appendix"].copy(), SETTING_KEYS)
    return mainboard_df, boundary_df, appendix_df


def build_summary_markdown(
    *,
    merged_results_online_df: pd.DataFrame,
    recomputed_results_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    arg_df: pd.DataFrame,
    wgr_df: pd.DataFrame,
    ri_df: pd.DataFrame,
    mainboard_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
    appendix_df: pd.DataFrame,
    merged_window_errors_out: Path | None,
    summary_title: str,
) -> str:
    if "n_valid_windows" in diagnostics_df.columns:
        zero_valid_df = diagnostics_df.loc[diagnostics_df["n_valid_windows"].fillna(0).astype(int) <= 0].copy()
    else:
        zero_valid_df = pd.DataFrame(columns=diagnostics_df.columns)
    lines = [
        f"# {summary_title}",
        "",
        "最终表格以 `window_errors.csv` 的 finite-window recomputation 为准。训练阶段在线聚合只用于监控，不作为最终论文表格依据。",
        "",
        "## Overview",
        "",
        f"- merged setting-level online rows: {len(merged_results_online_df)}",
        f"- recomputed setting rows: {len(recomputed_results_df)}",
        f"- zero-valid settings: {len(zero_valid_df)}",
        f"- ARG rows: {len(arg_df)}",
        f"- WGR rows: {len(wgr_df)}",
        f"- RI rows: {len(ri_df)}",
        f"- clean mainboard rows: {len(mainboard_df)}",
        f"- boundary board rows: {len(boundary_df)}",
        f"- appendix/failure rows: {len(appendix_df)}",
    ]
    if merged_window_errors_out is not None:
        lines.append(f"- merged window_errors path: `{merged_window_errors_out}`")

    lines.extend(["", "## Zero-Valid Settings", ""])
    if zero_valid_df.empty:
        lines.append("- 无 `n_valid_windows=0` 的 setting。")
    else:
        for row in zero_valid_df.sort_values(SETTING_KEYS).itertuples(index=False):
            lines.append(
                f"- {row.dataset_name} / L{int(row.lookback)} / H{int(row.horizon)} / {row.eval_view_name} / seed{int(row.seed)}"
            )

    lines.extend(["", "## Board Notes", ""])
    lines.append("- clean mainboard: ETTh1, ETTm1, exchange_rate, weather, solar_AL")
    lines.append("- boundary board: ETTh2")
    lines.append("- appendix / robustness / failure: ETTm2, electricity, 以及任何 `n_valid_windows=0` 的 setting")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    window_error_paths = [Path(path) for path in args.window_errors]
    results_online_paths = [Path(path) for path in args.results_online]
    merged_results_out = Path(args.merged_results_out)
    merged_window_errors_out = Path(args.merged_window_errors_out) if str(args.merged_window_errors_out).strip() else None
    results_out = Path(args.results_out)
    diagnostics_out = Path(args.diagnostics_out)
    arg_out = Path(args.arg_out)
    wgr_out = Path(args.wgr_out)
    ri_out = Path(args.ri_out)
    mainboard_out = Path(args.mainboard_out)
    boundary_board_out = Path(args.boundary_board_out)
    appendix_board_out = Path(args.appendix_board_out)
    summary_out = Path(args.summary_out)

    for path in [
        merged_results_out,
        results_out,
        diagnostics_out,
        arg_out,
        wgr_out,
        ri_out,
        mainboard_out,
        boundary_board_out,
        appendix_board_out,
        summary_out,
    ]:
        ensure_parent(path)
    if merged_window_errors_out is not None:
        ensure_parent(merged_window_errors_out)

    merged_results_online_df = read_results_online(results_online_paths)
    diagnostics, setting_metrics, group_metrics = aggregate_window_errors(
        window_error_paths=window_error_paths,
        merged_window_errors_out=merged_window_errors_out,
        chunksize=args.chunksize,
    )

    result_keys = []
    if not merged_results_online_df.empty and set(SETTING_KEYS).issubset(merged_results_online_df.columns):
        result_keys = [row_to_setting_key(record, SETTING_KEYS) for record in merged_results_online_df.to_dict(orient="records")]
    all_keys = sorted(set(result_keys) | set(diagnostics.keys()))

    diagnostics_df = build_diagnostics_frame(all_keys, diagnostics)
    recomputed_results_df = build_recomputed_results(merged_results_online_df, diagnostics_df, setting_metrics)
    arg_df = compute_aif_arg_table(recomputed_results_df)
    wgr_df = build_wgr_frame(diagnostics_df, setting_metrics, group_metrics)
    ri_df = compute_aif_ri_table(recomputed_results_df)
    mainboard_df, boundary_df, appendix_df = build_board_frames(recomputed_results_df)

    merged_results_online_df.to_csv(merged_results_out, index=False)
    recomputed_results_df.to_csv(results_out, index=False)
    diagnostics_df.to_csv(diagnostics_out, index=False)
    arg_df.to_csv(arg_out, index=False)
    wgr_df.to_csv(wgr_out, index=False)
    ri_df.to_csv(ri_out, index=False)
    mainboard_df.to_csv(mainboard_out, index=False)
    boundary_df.to_csv(boundary_board_out, index=False)
    appendix_df.to_csv(appendix_board_out, index=False)
    summary_out.write_text(
        build_summary_markdown(
            merged_results_online_df=merged_results_online_df,
            recomputed_results_df=recomputed_results_df,
            diagnostics_df=diagnostics_df,
            arg_df=arg_df,
            wgr_df=wgr_df,
            ri_df=ri_df,
            mainboard_df=mainboard_df,
            boundary_df=boundary_df,
            appendix_df=appendix_df,
            merged_window_errors_out=merged_window_errors_out,
            summary_title=str(args.summary_title or "AIF-Plus Summary"),
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
