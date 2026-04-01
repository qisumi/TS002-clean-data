from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data import ROOT_DIR, write_markdown


RAW_BOARD = "legacy_raw"
CLEAN_BOARD = "qc_clean"
BOUNDARY_BOARD = "support_boundary"
APPENDIX_BOARD = "clean_appendix"
ROBUST_BOARD = "counterfactual_robustness"
UPPER_BOUND_BOARD = "exploit_upper_bound"

PLANNED_MAINBOARD_DATASETS = {
    "ETTh1",
    "ETTm1",
    "exchange_rate",
    "weather",
    "electricity",
    "solar_AL",
}
PLANNED_BOUNDARY_DATASETS = {"ETTh2"}
PLANNED_APPENDIX_DATASETS = {"ETTm2"}

LEADERBOARD_SUMMARY_COLUMNS = [
    "board",
    "method",
    "method_family",
    "n_tasks",
    "mean_mae",
    "mean_mse",
    "mean_smape",
    "average_rank",
    "median_rank",
    "win_count",
]

SIGNIFICANCE_COLUMNS = [
    "board",
    "challenger",
    "reference",
    "n_common_tasks",
    "wins",
    "losses",
    "ties",
    "win_rate",
    "mean_delta_mae",
    "median_delta_mae",
    "sign_test_pvalue",
    "bootstrap_n_windows",
    "bootstrap_mean_delta",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
]

ERROR_DISTRIBUTION_COLUMNS = [
    "board",
    "dataset_name",
    "lookback",
    "horizon",
    "method",
    "eval_view_name",
    "n_windows",
    "mean_mae",
    "median_mae",
    "p75_mae",
    "p90_mae",
    "p95_mae",
    "p99_mae",
    "tail_mean_p95",
    "flagged_mae",
    "clean_target_mae",
    "flagged_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified leaderboard appendix and P2 analysis tables.")
    parser.add_argument("--results-dir", default=str(Path("results")))
    parser.add_argument("--reports-dir", default=str(Path("reports")))
    parser.add_argument("--stats-dir", default=str(Path("statistic_results")))
    parser.add_argument("--support-summary", default=str(Path("reports") / "clean_view_support_summary.csv"))
    parser.add_argument("--out-md", default=str(Path("reports") / "unified_leaderboard_appendix.md"))
    parser.add_argument("--rows-out", default=str(Path("results") / "unified_leaderboard_rows.csv"))
    parser.add_argument("--summary-out", default=str(Path("results") / "unified_leaderboard_summary.csv"))
    parser.add_argument("--significance-out", default=str(Path("results") / "unified_significance.csv"))
    parser.add_argument("--error-out", default=str(Path("results") / "unified_error_distribution.csv"))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--min-common-tasks", type=int, default=3)
    return parser.parse_args()


def read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def ensure_nullable_int(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        return pd.Series(pd.array([pd.NA] * len(series), dtype="Int64"), index=series.index)
    return numeric.round().astype("Int64")


def fill_missing_lookback(df: pd.DataFrame, support_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "lookback" not in out.columns:
        out["lookback"] = pd.Series(pd.array([pd.NA] * len(out), dtype="Int64"))
    else:
        out["lookback"] = ensure_nullable_int(out["lookback"])

    if support_df.empty or "dataset_name" not in out.columns or "horizon" not in out.columns:
        return out

    mapping = (
        support_df[["dataset_name", "horizon", "lookback"]]
        .dropna(subset=["dataset_name", "horizon", "lookback"])
        .drop_duplicates()
    )
    counts = mapping.groupby(["dataset_name", "horizon"], dropna=False).size().reset_index(name="n")
    mapping = mapping.merge(counts, on=["dataset_name", "horizon"], how="left")
    mapping = mapping[mapping["n"] == 1].drop(columns=["n"])
    if mapping.empty:
        return out

    missing = out["lookback"].isna()
    if not bool(missing.any()):
        return out

    fill_df = out.loc[missing, :].merge(mapping, on=["dataset_name", "horizon"], how="left", suffixes=("", "_support"))
    if "lookback_support" not in fill_df.columns:
        return out
    fill_values = ensure_nullable_int(fill_df["lookback_support"])
    out.loc[missing, "lookback"] = fill_values.to_numpy()
    return out


def build_support_status(support_df: pd.DataFrame, clean_rows_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if support_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset_name",
                "lookback",
                "horizon",
                "clean_board_status",
                "clean_eval_available",
                "train_clean_ratio",
                "val_clean_windows",
                "test_clean_windows",
            ]
        )

    work = support_df.copy()
    work["lookback"] = ensure_nullable_int(work["lookback"])
    work["horizon"] = ensure_nullable_int(work["horizon"])
    work["clean_windows"] = np.maximum(
        pd.to_numeric(work.get("conservative_clean_windows", 0), errors="coerce").fillna(0.0),
        pd.to_numeric(work.get("phase_balanced_windows", 0), errors="coerce").fillna(0.0),
    )
    work["raw_windows"] = pd.to_numeric(work.get("raw_windows", 0), errors="coerce").fillna(0.0)
    work["clean_ratio"] = work["clean_windows"] / work["raw_windows"].clip(lower=1.0)

    pivot = (
        work.pivot_table(
            index=["dataset_name", "lookback", "horizon"],
            columns="split_name",
            values=["clean_windows", "clean_ratio", "raw_windows"],
            aggfunc="first",
        )
        .sort_index(axis=1)
        .reset_index()
    )
    pivot.columns = [
        "_".join([str(part) for part in col if str(part) != ""]).strip("_") if isinstance(col, tuple) else str(col)
        for col in pivot.columns
    ]

    clean_eval_keys: set[tuple[str, int | None, int | None]] = set()
    if clean_rows_df is not None and not clean_rows_df.empty:
        clean_subset = clean_rows_df.copy()
        clean_subset["lookback"] = ensure_nullable_int(clean_subset.get("lookback", pd.Series([pd.NA] * len(clean_subset))))
        clean_subset["horizon"] = ensure_nullable_int(clean_subset.get("horizon", pd.Series([pd.NA] * len(clean_subset))))
        for row in clean_subset[["dataset_name", "lookback", "horizon"]].drop_duplicates().itertuples(index=False):
            lookback = None if pd.isna(row.lookback) else int(row.lookback)
            horizon = None if pd.isna(row.horizon) else int(row.horizon)
            clean_eval_keys.add((str(row.dataset_name), lookback, horizon))

    rows: list[dict[str, Any]] = []
    def _safe_count(value: Any) -> int:
        if pd.isna(value):
            return 0
        return int(float(value))

    for row in pivot.itertuples(index=False):
        dataset_name = str(row.dataset_name)
        lookback = row.lookback
        horizon = row.horizon
        train_ratio = float(getattr(row, "clean_ratio_train", np.nan))
        val_clean = _safe_count(getattr(row, "clean_windows_val", 0))
        test_clean = _safe_count(getattr(row, "clean_windows_test", 0))
        lookback_key = None if pd.isna(lookback) else int(lookback)
        horizon_key = None if pd.isna(horizon) else int(horizon)
        clean_eval_available = (dataset_name, lookback_key, horizon_key) in clean_eval_keys
        has_val_or_test_support = val_clean > 0 or test_clean > 0

        if dataset_name in PLANNED_BOUNDARY_DATASETS:
            status = "boundary"
        elif dataset_name in PLANNED_APPENDIX_DATASETS:
            status = "appendix"
        elif dataset_name in PLANNED_MAINBOARD_DATASETS:
            status = "mainboard" if has_val_or_test_support or clean_eval_available else "appendix"
        elif has_val_or_test_support:
            status = "boundary"
        else:
            status = "appendix"

        rows.append(
            {
                "dataset_name": dataset_name,
                "lookback": lookback,
                "horizon": horizon,
                "clean_board_status": status,
                "clean_eval_available": int(clean_eval_available),
                "train_clean_ratio": train_ratio,
                "val_clean_windows": val_clean,
                "test_clean_windows": test_clean,
            }
        )
    return pd.DataFrame(rows)


def format_method_name(prefix: str, backbone: str | None = None) -> str:
    token = str(prefix)
    if backbone is None or not str(backbone).strip():
        return token
    return f"{token}/{str(backbone).strip()}"


def _leaderboard_role(method_family: str) -> str:
    if method_family.startswith("AEF"):
        return "upper_bound_only"
    return "leaderboard"


def standardize_board_rows(
    df: pd.DataFrame,
    *,
    board: str,
    source_name: str,
    method_series: pd.Series,
    family_name: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "dataset_name",
                "lookback",
                "horizon",
                "method",
                "method_family",
                "leaderboard_role",
                "source_name",
                "board",
                "train_view_name",
                "eval_view_name",
                "mae",
                "mse",
                "smape",
                "seed",
            ]
        )

    out = df.copy()
    out["lookback"] = ensure_nullable_int(out.get("lookback", pd.Series([pd.NA] * len(out))))
    out["horizon"] = ensure_nullable_int(out["horizon"])
    out["seed"] = ensure_nullable_int(out.get("seed", pd.Series([0] * len(out))))
    out["method"] = method_series.astype(str)
    out["method_family"] = family_name
    out["leaderboard_role"] = _leaderboard_role(family_name)
    out["source_name"] = source_name
    out["board"] = board
    for metric in ["mae", "mse", "smape"]:
        out[metric] = pd.to_numeric(out.get(metric), errors="coerce")
    return out[
        [
            "dataset_name",
            "lookback",
            "horizon",
            "method",
            "method_family",
            "leaderboard_role",
            "source_name",
            "board",
            "train_view_name",
            "eval_view_name",
            "mae",
            "mse",
            "smape",
            "seed",
        ]
    ].copy()


def load_counterfactual_boards(counterfactual_df: pd.DataFrame) -> pd.DataFrame:
    if counterfactual_df.empty:
        return pd.DataFrame()
    completed = counterfactual_df[
        counterfactual_df.get("status", "").astype(str).eq("completed")
        & counterfactual_df.get("eval_protocol", "").astype(str).eq("view_matrix")
        & counterfactual_df.get("subset_name", "").astype(str).eq("full_view")
    ].copy()
    if completed.empty:
        return completed

    raw_rows = completed[
        completed["train_view_name"].astype(str).eq("raw")
        & completed["eval_view_name"].astype(str).eq("raw")
    ].copy()
    clean_rows = completed[
        ~completed["eval_view_name"].astype(str).isin(["raw", "intervened"])
        & completed["train_view_name"].astype(str).eq(completed["eval_view_name"].astype(str))
    ].copy()
    return pd.concat(
        [
            standardize_board_rows(
                raw_rows,
                board=RAW_BOARD,
                source_name="counterfactual_erm",
                method_series=raw_rows["backbone"],
                family_name="ERM",
            ),
            standardize_board_rows(
                clean_rows,
                board=CLEAN_BOARD,
                source_name="counterfactual_erm",
                method_series=clean_rows["backbone"],
                family_name="ERM",
            ),
        ],
        ignore_index=True,
    )


def load_aif_plus_boards(aif_plus_df: pd.DataFrame) -> pd.DataFrame:
    if aif_plus_df.empty:
        return pd.DataFrame()
    raw_rows = aif_plus_df[aif_plus_df["eval_view_name"].astype(str).eq("raw")].copy()
    clean_rows = aif_plus_df[~aif_plus_df["eval_view_name"].astype(str).isin(["raw", "intervened"])].copy()
    return pd.concat(
        [
            standardize_board_rows(
                raw_rows,
                board=RAW_BOARD,
                source_name="aif_plus",
                method_series=pd.Series(["AIF-Plus"] * len(raw_rows), index=raw_rows.index),
                family_name="AIF-Plus",
            ),
            standardize_board_rows(
                clean_rows,
                board=CLEAN_BOARD,
                source_name="aif_plus",
                method_series=pd.Series(["AIF-Plus"] * len(clean_rows), index=clean_rows.index),
                family_name="AIF-Plus",
            ),
        ],
        ignore_index=True,
    )


def load_upper_bound_rows(
    aef_plus_df: pd.DataFrame,
    aef_df: pd.DataFrame,
    aef_control_df: pd.DataFrame,
) -> pd.DataFrame:
    frames = [
        ("AEF-Plus", "aef_plus", aef_plus_df),
        ("AEF-Weak", "aef_baseline", aef_df),
        ("AEF-Control", "aef_control", aef_control_df),
    ]
    outputs: list[pd.DataFrame] = []
    for method_name, source_name, df in frames:
        if df.empty:
            continue
        outputs.append(
            standardize_board_rows(
                df,
                board=UPPER_BOUND_BOARD,
                source_name=source_name,
                method_series=pd.Series([method_name] * len(df), index=df.index),
                family_name=method_name,
            )
        )
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def attach_board_status(rows_df: pd.DataFrame, support_status_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return rows_df.copy()
    out = rows_df.copy()
    if support_status_df.empty:
        out["clean_board_status"] = "unknown"
        return out
    merged = out.merge(
        support_status_df,
        on=["dataset_name", "lookback", "horizon"],
        how="left",
    )
    merged["clean_board_status"] = merged["clean_board_status"].fillna("unknown")
    clean_mask = merged["board"].astype(str).eq(CLEAN_BOARD)
    boundary_mask = clean_mask & merged["clean_board_status"].astype(str).eq("boundary")
    appendix_mask = clean_mask & ~merged["clean_board_status"].astype(str).isin(["mainboard", "boundary"])
    merged.loc[boundary_mask, "board"] = BOUNDARY_BOARD
    merged.loc[appendix_mask, "board"] = APPENDIX_BOARD
    return merged


def assign_board_ranks(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return rows_df.copy()
    out = rows_df.copy()
    out["task_key"] = out.apply(
        lambda row: f"{row['dataset_name']}|L{row['lookback']}|H{row['horizon']}" if pd.notna(row["lookback"]) else f"{row['dataset_name']}|H{row['horizon']}",
        axis=1,
    )
    out["rank_mae"] = (
        out.groupby(["board", "dataset_name", "lookback", "horizon"], dropna=False)["mae"]
        .rank(method="min", ascending=True)
    )
    out["is_win"] = out["rank_mae"].eq(1).astype(int)
    return out


def summarize_leaderboards(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame(columns=LEADERBOARD_SUMMARY_COLUMNS)
    eligible = rows_df[rows_df["leaderboard_role"].astype(str).eq("leaderboard")].copy()
    if eligible.empty:
        return pd.DataFrame(columns=LEADERBOARD_SUMMARY_COLUMNS)
    summary = (
        eligible.groupby(["board", "method", "method_family"], dropna=False)
        .agg(
            n_tasks=("mae", "size"),
            mean_mae=("mae", "mean"),
            mean_mse=("mse", "mean"),
            mean_smape=("smape", "mean"),
            average_rank=("rank_mae", "mean"),
            median_rank=("rank_mae", "median"),
            win_count=("is_win", "sum"),
        )
        .reset_index()
    )
    return summary.sort_values(["board", "average_rank", "mean_mae", "method"]).reset_index(drop=True)


def exact_sign_test_pvalue(wins: int, losses: int) -> float:
    n = int(wins + losses)
    if n <= 0:
        return float("nan")
    k = min(int(wins), int(losses))
    tail_prob = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return float(min(1.0, 2.0 * tail_prob))


def build_task_level_significance(rows_df: pd.DataFrame, min_common_tasks: int) -> pd.DataFrame:
    eligible = rows_df[rows_df["leaderboard_role"].astype(str).eq("leaderboard")].copy()
    if eligible.empty:
        return pd.DataFrame(columns=SIGNIFICANCE_COLUMNS)

    results: list[dict[str, Any]] = []
    for board, board_df in eligible.groupby("board", dropna=False):
        methods = sorted(board_df["method"].astype(str).unique().tolist())
        for challenger, reference in combinations(methods, 2):
            pair_df = board_df[board_df["method"].astype(str).isin([challenger, reference])].copy()
            pivot = (
                pair_df.pivot_table(
                    index=["dataset_name", "lookback", "horizon"],
                    columns="method",
                    values="mae",
                    aggfunc="mean",
                )
                .dropna()
                .reset_index()
            )
            if len(pivot) < min_common_tasks:
                continue
            diffs = (pivot[challenger] - pivot[reference]).to_numpy(dtype=float)
            wins = int((diffs < -1e-12).sum())
            losses = int((diffs > 1e-12).sum())
            ties = int(len(diffs) - wins - losses)
            results.append(
                {
                    "board": board,
                    "challenger": challenger,
                    "reference": reference,
                    "n_common_tasks": int(len(diffs)),
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "win_rate": float(wins / max(wins + losses, 1)),
                    "mean_delta_mae": float(np.mean(diffs)),
                    "median_delta_mae": float(np.median(diffs)),
                    "sign_test_pvalue": exact_sign_test_pvalue(wins, losses),
                    "bootstrap_n_windows": pd.NA,
                    "bootstrap_mean_delta": pd.NA,
                    "bootstrap_ci_low": pd.NA,
                    "bootstrap_ci_high": pd.NA,
                }
            )
    return pd.DataFrame(results, columns=SIGNIFICANCE_COLUMNS)


def standardize_window_errors(
    df: pd.DataFrame,
    *,
    source_name: str,
    method_resolver: Any,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["lookback"] = ensure_nullable_int(out.get("lookback", pd.Series([pd.NA] * len(out))))
    out["horizon"] = ensure_nullable_int(out["horizon"])
    out["method"] = method_resolver(out)
    out["source_name"] = source_name
    out["mae"] = pd.to_numeric(out.get("mae"), errors="coerce")
    out["is_flagged"] = ensure_nullable_int(out.get("is_flagged", pd.Series([pd.NA] * len(out))))
    out["strict_target_clean"] = ensure_nullable_int(out.get("strict_target_clean", pd.Series([pd.NA] * len(out))))
    out["board"] = np.where(
        out["eval_view_name"].astype(str).eq("raw"),
        RAW_BOARD,
        np.where(
            out["eval_view_name"].astype(str).eq("intervened"),
            ROBUST_BOARD,
            CLEAN_BOARD,
        ),
    )
    return out


def bootstrap_mean_diff(diff: np.ndarray, samples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    n = len(diff)
    for idx in range(samples):
        picked = rng.integers(0, n, size=n)
        means[idx] = float(diff[picked].mean())
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def attach_window_bootstrap(
    significance_df: pd.DataFrame,
    counterfactual_window_df: pd.DataFrame,
    aif_plus_window_df: pd.DataFrame,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if significance_df.empty:
        return significance_df.copy()

    frames: list[pd.DataFrame] = []
    if not counterfactual_window_df.empty:
        frames.append(
            standardize_window_errors(
                counterfactual_window_df,
                source_name="counterfactual_window_errors",
                method_resolver=lambda df: df["backbone"].astype(str),
            )
        )
    if not aif_plus_window_df.empty:
        frames.append(
            standardize_window_errors(
                aif_plus_window_df,
                source_name="aif_plus_window_errors",
                method_resolver=lambda df: pd.Series(["AIF-Plus"] * len(df), index=df.index),
            )
        )
    if not frames:
        return significance_df.copy()

    errors_df = pd.concat(frames, ignore_index=True)
    out = significance_df.copy()
    for row in out.itertuples(index=False):
        if not (
            str(row.challenger).startswith("AIF")
            or str(row.reference).startswith("AIF")
        ):
            continue
        subset = errors_df[
            errors_df["board"].astype(str).eq(str(row.board))
            & errors_df["method"].astype(str).isin([str(row.challenger), str(row.reference)])
        ].copy()
        if subset.empty:
            continue
        pivot = subset.pivot_table(
            index=["dataset_name", "lookback", "horizon", "eval_view_name", "window_id"],
            columns="method",
            values="mae",
            aggfunc="mean",
        ).dropna()
        if pivot.empty:
            continue
        diff = (pivot[str(row.challenger)] - pivot[str(row.reference)]).to_numpy(dtype=float)
        mean_delta, ci_low, ci_high = bootstrap_mean_diff(diff, samples=bootstrap_samples, seed=bootstrap_seed)
        mask = (
            out["board"].astype(str).eq(str(row.board))
            & out["challenger"].astype(str).eq(str(row.challenger))
            & out["reference"].astype(str).eq(str(row.reference))
        )
        out.loc[mask, "bootstrap_n_windows"] = int(len(diff))
        out.loc[mask, "bootstrap_mean_delta"] = mean_delta
        out.loc[mask, "bootstrap_ci_low"] = ci_low
        out.loc[mask, "bootstrap_ci_high"] = ci_high
    return out


def build_error_distribution_summary(
    counterfactual_window_df: pd.DataFrame,
    aif_plus_window_df: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not counterfactual_window_df.empty:
        frames.append(
            standardize_window_errors(
                counterfactual_window_df,
                source_name="counterfactual_window_errors",
                method_resolver=lambda df: df["backbone"].astype(str),
            )
        )
    if not aif_plus_window_df.empty:
        frames.append(
            standardize_window_errors(
                aif_plus_window_df,
                source_name="aif_plus_window_errors",
                method_resolver=lambda df: pd.Series(["AIF-Plus"] * len(df), index=df.index),
            )
        )
    if not frames:
        return pd.DataFrame(columns=ERROR_DISTRIBUTION_COLUMNS)

    errors_df = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, Any]] = []
    group_cols = ["board", "dataset_name", "lookback", "horizon", "method", "eval_view_name"]
    for key, group in errors_df.groupby(group_cols, dropna=False):
        mae = group["mae"].dropna().to_numpy(dtype=float)
        if len(mae) == 0:
            continue
        p95 = float(np.quantile(mae, 0.95))
        flagged_mask = group["is_flagged"].fillna(0).astype(int).eq(1)
        clean_mask = group["strict_target_clean"].fillna(0).astype(int).eq(1)
        rows.append(
            {
                "board": key[0],
                "dataset_name": key[1],
                "lookback": key[2],
                "horizon": key[3],
                "method": key[4],
                "eval_view_name": key[5],
                "n_windows": int(len(mae)),
                "mean_mae": float(np.mean(mae)),
                "median_mae": float(np.median(mae)),
                "p75_mae": float(np.quantile(mae, 0.75)),
                "p90_mae": float(np.quantile(mae, 0.90)),
                "p95_mae": p95,
                "p99_mae": float(np.quantile(mae, 0.99)),
                "tail_mean_p95": float(mae[mae >= p95].mean()),
                "flagged_mae": float(group.loc[flagged_mask, "mae"].mean()) if bool(flagged_mask.any()) else np.nan,
                "clean_target_mae": float(group.loc[clean_mask, "mae"].mean()) if bool(clean_mask.any()) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=ERROR_DISTRIBUTION_COLUMNS)
    out["flagged_gap"] = out["flagged_mae"] - out["clean_target_mae"]
    return out.sort_values(["board", "dataset_name", "horizon", "mean_mae", "method"]).reset_index(drop=True)


def build_robustness_summary(
    arg_df: pd.DataFrame,
    wgr_df: pd.DataFrame,
    aif_plus_arg_df: pd.DataFrame,
    aif_plus_wgr_df: pd.DataFrame,
) -> pd.DataFrame:
    arg_rows: list[pd.DataFrame] = []
    if not arg_df.empty:
        base = arg_df[arg_df.get("scope", "").astype(str).eq("overall")].copy()
        if not base.empty:
            base["method"] = base["backbone"].astype(str)
            arg_rows.append(base[["dataset_name", "lookback", "horizon", "method", "ARG_mae"]])
    if not aif_plus_arg_df.empty:
        plus = aif_plus_arg_df.copy()
        plus["method"] = "AIF-Plus"
        arg_rows.append(plus[["dataset_name", "lookback", "horizon", "method", "ARG_mae"]])

    wgr_rows: list[pd.DataFrame] = []
    if not wgr_df.empty:
        base = wgr_df[wgr_df.get("status", "").astype(str).eq("completed")].copy()
        if not base.empty:
            base["method"] = base["backbone"].astype(str)
            wgr_rows.append(base[["dataset_name", "lookback", "horizon", "method", "WGR_gap"]])
    if not aif_plus_wgr_df.empty:
        plus = aif_plus_wgr_df.copy()
        plus["method"] = "AIF-Plus"
        wgr_rows.append(plus[["dataset_name", "lookback", "horizon", "method", "WGR_gap"]])

    arg_table = pd.concat(arg_rows, ignore_index=True) if arg_rows else pd.DataFrame()
    wgr_table = pd.concat(wgr_rows, ignore_index=True) if wgr_rows else pd.DataFrame()
    if arg_table.empty and wgr_table.empty:
        return pd.DataFrame()

    merged = None
    if not arg_table.empty:
        arg_table = arg_table.copy()
        arg_table["lookback"] = ensure_nullable_int(arg_table["lookback"])
        arg_table["horizon"] = ensure_nullable_int(arg_table["horizon"])
        merged = arg_table
    if not wgr_table.empty:
        wgr_table = wgr_table.copy()
        wgr_table["lookback"] = ensure_nullable_int(wgr_table["lookback"])
        wgr_table["horizon"] = ensure_nullable_int(wgr_table["horizon"])
        if merged is None:
            merged = wgr_table
        else:
            merged = merged.merge(
                wgr_table,
                on=["dataset_name", "lookback", "horizon", "method"],
                how="outer",
            )
    assert merged is not None
    summary = (
        merged.groupby("method", dropna=False)
        .agg(
            n_tasks=("dataset_name", "size"),
            mean_ARG_mae=("ARG_mae", "mean"),
            mean_WGR_gap=("WGR_gap", "mean"),
        )
        .reset_index()
    )
    summary["ARG_rank"] = summary["mean_ARG_mae"].rank(method="min", ascending=True)
    summary["WGR_rank"] = summary["mean_WGR_gap"].rank(method="min", ascending=True)
    summary["robustness_rank"] = summary[["ARG_rank", "WGR_rank"]].mean(axis=1)
    return summary.sort_values(["robustness_rank", "mean_ARG_mae", "method"]).reset_index(drop=True)


def build_upper_bound_summary(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame()
    return (
        rows_df.groupby(["method", "eval_view_name"], dropna=False)
        .agg(
            n_tasks=("mae", "size"),
            mean_mae=("mae", "mean"),
            mean_mse=("mse", "mean"),
            mean_smape=("smape", "mean"),
        )
        .reset_index()
        .sort_values(["eval_view_name", "mean_mae", "method"])
        .reset_index(drop=True)
    )


def fmt_lb_h(dataset_name: str, lookback: Any, horizon: Any) -> str:
    if pd.notna(lookback):
        return f"{dataset_name} / L{int(lookback)} / H{int(horizon)}"
    return f"{dataset_name} / H{int(horizon)}"


def build_markdown(
    board_summary_df: pd.DataFrame,
    board_rows_df: pd.DataFrame,
    significance_df: pd.DataFrame,
    error_df: pd.DataFrame,
    robustness_summary_df: pd.DataFrame,
    support_status_df: pd.DataFrame,
    upper_bound_df: pd.DataFrame,
) -> str:
    lines = [
        "# Unified Leaderboard Appendix",
        "",
        "这页把 `Legacy-Raw Board / QC-Clean Mainboard / Counterfactual Robustness Board` 统一收口，并补上 P2 所需的 task-level significance 与窗口级误差分布摘要。",
        "",
        "## Coverage",
        "",
    ]

    if support_status_df.empty:
        lines.append("- clean support summary 缺失，无法标注 mainboard/boundary 状态。")
    else:
        status_counts = support_status_df["clean_board_status"].astype(str).value_counts().to_dict()
        completed_counts = support_status_df["clean_eval_available"].astype(int).value_counts().to_dict()
        lines.append(f"- clean board 状态分布: {status_counts}")
        lines.append(f"- clean eval availability: {completed_counts}")
        for row in support_status_df.sort_values(["dataset_name", "lookback", "horizon"]).itertuples(index=False):
            lines.append(
                f"- {fmt_lb_h(row.dataset_name, row.lookback, row.horizon)}: "
                f"{row.clean_board_status}, train_clean_ratio={float(row.train_clean_ratio):.3f}, "
                f"val_clean_windows={int(row.val_clean_windows)}, test_clean_windows={int(row.test_clean_windows)}, "
                f"clean_eval_available={int(row.clean_eval_available)}"
            )

    lines.extend(["", "## Unified Boards", ""])
    for board_name, title in [
        (RAW_BOARD, "Legacy-Raw Board"),
        (CLEAN_BOARD, "QC-Clean Mainboard"),
        (BOUNDARY_BOARD, "Support-Boundary Board"),
        (APPENDIX_BOARD, "Appendix-Only Clean Tasks"),
    ]:
        lines.extend([f"### {title}", ""])
        subset = board_summary_df[board_summary_df["board"].astype(str).eq(board_name)].copy() if not board_summary_df.empty else pd.DataFrame()
        if subset.empty:
            lines.append("- 当前没有可用结果。")
            lines.append("")
            continue
        for row in subset.head(12).itertuples(index=False):
            lines.append(
                f"- {row.method}: average_rank={float(row.average_rank):.3f}, "
                f"mean_MAE={float(row.mean_mae):.4f}, wins={int(row.win_count)}/{int(row.n_tasks)}"
            )
        lines.append("")

    lines.extend(["## Robustness Board", ""])
    if robustness_summary_df.empty:
        lines.append("- 当前没有可聚合的 ARG/WGR 结果。")
    else:
        for row in robustness_summary_df.head(12).itertuples(index=False):
            lines.append(
                f"- {row.method}: robustness_rank={float(row.robustness_rank):.3f}, "
                f"mean_ARG={float(row.mean_ARG_mae):.4f}, mean_WGR_gap={float(row.mean_WGR_gap):.4f}"
            )

    lines.extend(["", "## Significance", ""])
    if significance_df.empty:
        lines.append("- 当前没有满足最小公共 task 数的成对比较。")
    else:
        for board_name in [RAW_BOARD, CLEAN_BOARD, BOUNDARY_BOARD, APPENDIX_BOARD]:
            subset = significance_df[significance_df["board"].astype(str).eq(board_name)].copy()
            if subset.empty:
                continue
            lines.append(f"### {board_name}")
            lines.append("")
            priority = subset[
                subset["challenger"].astype(str).str.contains("AIF", regex=False)
                | subset["reference"].astype(str).str.contains("AIF", regex=False)
            ].copy()
            display = priority if not priority.empty else subset
            display = display.sort_values(["sign_test_pvalue", "mean_delta_mae"]).head(10)
            for row in display.itertuples(index=False):
                bootstrap_note = ""
                if pd.notna(row.bootstrap_ci_low) and pd.notna(row.bootstrap_ci_high):
                    bootstrap_note = (
                        f", bootstrap_mean_delta={float(row.bootstrap_mean_delta):.4f}, "
                        f"95%CI=[{float(row.bootstrap_ci_low):.4f}, {float(row.bootstrap_ci_high):.4f}]"
                    )
                lines.append(
                    f"- {row.challenger} vs {row.reference}: n_tasks={int(row.n_common_tasks)}, "
                    f"wins={int(row.wins)}, losses={int(row.losses)}, mean_delta={float(row.mean_delta_mae):.4f}, "
                    f"sign_test_p={float(row.sign_test_pvalue):.4g}{bootstrap_note}"
                )
            lines.append("")

    lines.extend(["## Error Distribution", ""])
    if error_df.empty:
        lines.append("- 当前没有窗口级误差长表，无法给出分位数与 tail summary。")
    else:
        priority_methods = sorted(
            {
                str(item)
                for item in error_df["method"].astype(str).tolist()
                if item.startswith("AIF") or item in {"DLinear", "PatchTST", "iTransformer", "ModernTCN", "TimeMixer", "TimeMixerPP", "TQNet"}
            }
        )
        shown = error_df[error_df["method"].astype(str).isin(priority_methods)].copy()
        shown = shown.sort_values(["board", "tail_mean_p95", "mean_mae"]).head(16)
        for row in shown.itertuples(index=False):
            flagged_gap_text = "NA" if pd.isna(row.flagged_gap) else f"{float(row.flagged_gap):.4f}"
            lines.append(
                f"- {fmt_lb_h(row.dataset_name, row.lookback, row.horizon)} / {row.method} / {row.eval_view_name}: "
                f"median={float(row.median_mae):.4f}, p90={float(row.p90_mae):.4f}, p95={float(row.p95_mae):.4f}, "
                f"tail_mean_p95={float(row.tail_mean_p95):.4f}, flagged_gap={flagged_gap_text}"
            )

    lines.extend(["", "## Exploit Upper Bound", ""])
    if upper_bound_df.empty:
        lines.append("- 当前没有 AEF/upper-bound 结果。")
    else:
        for row in upper_bound_df.head(12).itertuples(index=False):
            lines.append(
                f"- {row.method} / {row.eval_view_name}: mean_MAE={float(row.mean_mae):.4f}, "
                f"mean_MSE={float(row.mean_mse):.4f}, tasks={int(row.n_tasks)}"
            )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `results/unified_leaderboard_rows.csv`",
            "- `results/unified_leaderboard_summary.csv`",
            "- `results/unified_significance.csv`",
            "- `results/unified_error_distribution.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results_dir = ROOT_DIR / Path(args.results_dir)
    reports_dir = ROOT_DIR / Path(args.reports_dir)
    support_path = ROOT_DIR / Path(args.support_summary)

    counterfactual_df = fill_missing_lookback(read_optional_csv(results_dir / "counterfactual_2x2.csv"), read_optional_csv(support_path))
    arg_df = fill_missing_lookback(read_optional_csv(results_dir / "artifact_reliance_gap.csv"), read_optional_csv(support_path))
    wgr_df = fill_missing_lookback(read_optional_csv(results_dir / "worst_group_risk.csv"), read_optional_csv(support_path))
    aif_plus_df = fill_missing_lookback(read_optional_csv(results_dir / "aif_plus_results.csv"), read_optional_csv(support_path))
    aif_plus_arg_df = fill_missing_lookback(read_optional_csv(results_dir / "aif_plus_artifact_reliance_gap.csv"), read_optional_csv(support_path))
    aif_plus_wgr_df = fill_missing_lookback(read_optional_csv(results_dir / "aif_plus_worst_group_risk.csv"), read_optional_csv(support_path))
    aef_plus_df = fill_missing_lookback(read_optional_csv(results_dir / "aef_plus_results.csv"), read_optional_csv(support_path))
    aef_df = fill_missing_lookback(read_optional_csv(results_dir / "aef_results.csv"), read_optional_csv(support_path))
    aef_control_df = fill_missing_lookback(read_optional_csv(results_dir / "aef_control_results.csv"), read_optional_csv(support_path))

    support_df = read_optional_csv(support_path)
    board_rows_df = pd.concat(
        [
            load_counterfactual_boards(counterfactual_df),
            load_aif_plus_boards(aif_plus_df),
            load_upper_bound_rows(
                aef_plus_df=aef_plus_df,
                aef_df=aef_df,
                aef_control_df=aef_control_df,
            ),
        ],
        ignore_index=True,
    )
    support_status_df = build_support_status(
        support_df,
        clean_rows_df=board_rows_df[board_rows_df["board"].astype(str).eq(CLEAN_BOARD)].copy(),
    )
    board_rows_df = attach_board_status(board_rows_df, support_status_df)
    board_rows_df = assign_board_ranks(board_rows_df)
    board_summary_df = summarize_leaderboards(board_rows_df)

    significance_df = build_task_level_significance(board_rows_df, min_common_tasks=int(args.min_common_tasks))
    significance_df = attach_window_bootstrap(
        significance_df=significance_df,
        counterfactual_window_df=fill_missing_lookback(read_optional_csv(results_dir / "counterfactual_window_errors.csv"), support_df),
        aif_plus_window_df=fill_missing_lookback(read_optional_csv(results_dir / "aif_plus_window_errors.csv"), support_df),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    error_df = build_error_distribution_summary(
        counterfactual_window_df=fill_missing_lookback(read_optional_csv(results_dir / "counterfactual_window_errors.csv"), support_df),
        aif_plus_window_df=fill_missing_lookback(read_optional_csv(results_dir / "aif_plus_window_errors.csv"), support_df),
    )
    robustness_summary_df = build_robustness_summary(
        arg_df=arg_df,
        wgr_df=wgr_df,
        aif_plus_arg_df=aif_plus_arg_df,
        aif_plus_wgr_df=aif_plus_wgr_df,
    )
    upper_bound_df = build_upper_bound_summary(board_rows_df[board_rows_df["board"].astype(str).eq(UPPER_BOUND_BOARD)].copy())

    rows_out = ROOT_DIR / Path(args.rows_out)
    summary_out = ROOT_DIR / Path(args.summary_out)
    significance_out = ROOT_DIR / Path(args.significance_out)
    error_out = ROOT_DIR / Path(args.error_out)
    out_md = ROOT_DIR / Path(args.out_md)
    for path in [rows_out, summary_out, significance_out, error_out, out_md]:
        path.parent.mkdir(parents=True, exist_ok=True)

    board_rows_df.to_csv(rows_out, index=False)
    board_summary_df.to_csv(summary_out, index=False)
    significance_df.to_csv(significance_out, index=False)
    error_df.to_csv(error_out, index=False)
    write_markdown(
        out_md,
        build_markdown(
            board_summary_df=board_summary_df,
            board_rows_df=board_rows_df,
            significance_df=significance_df,
            error_df=error_df,
            robustness_summary_df=robustness_summary_df,
            support_status_df=support_status_df,
            upper_bound_df=upper_bound_df,
        ),
    )


if __name__ == "__main__":
    main()
