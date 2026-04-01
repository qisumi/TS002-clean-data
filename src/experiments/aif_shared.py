from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_clean_view_name(config: dict[str, Any], dataset_name: str) -> str:
    return str(config["defaults"]["view_alias"][dataset_name]["clean_like"])


def build_lookup(values: list[str]) -> dict[str, int]:
    unique = sorted({value if value else "NA" for value in values} | {"NA"})
    return {name: idx for idx, name in enumerate(unique)}


def build_group_map(rows: pd.DataFrame) -> dict[str, int]:
    groups = sorted(rows["primary_group_key"].fillna("NA").astype(str).unique().tolist())
    return {group_name: idx for idx, group_name in enumerate(groups)}


def group_tail_mean(
    per_sample_loss: torch.Tensor,
    group_ids: torch.Tensor,
    tail_frac: float,
    round_up: bool = False,
) -> torch.Tensor:
    group_losses: list[torch.Tensor] = []
    for group_idx in torch.unique(group_ids):
        mask = group_ids == group_idx
        if bool(mask.any()):
            group_losses.append(per_sample_loss[mask].mean())
    if not group_losses:
        return per_sample_loss.mean()
    stacked = torch.stack(group_losses)
    raw_k = len(group_losses) * max(float(tail_frac), 1e-6)
    k = max(1, int(math.ceil(raw_k)) if round_up else int(raw_k))
    return torch.topk(stacked, k=k, largest=True).values.mean()


def build_loader(dataset: Dataset[Any], batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def compare_against_baseline(aif_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    if aif_df.empty or baseline_df.empty:
        return pd.DataFrame()
    baseline = baseline_df[(baseline_df["status"] == "completed") & (baseline_df["train_view_name"] == "raw")].copy()
    baseline = baseline.rename(columns={"mae": "baseline_mae", "mse": "baseline_mse", "smape": "baseline_smape"})
    keep_cols = ["dataset_name", "backbone", "horizon", "eval_view_name", "baseline_mae", "baseline_mse", "baseline_smape"]
    merge_cols = ["dataset_name", "backbone", "horizon", "eval_view_name"]
    if "lookback" in baseline.columns and "lookback" in aif_df.columns:
        keep_cols.insert(2, "lookback")
        merge_cols.insert(2, "lookback")
    return aif_df.merge(baseline[keep_cols], on=merge_cols, how="left")


def compute_aif_arg_table(results_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset_name",
        "backbone",
        "lookback",
        "horizon",
        "train_view_name",
        "mae_raw",
        "mae_intervened",
        "ARG_mae",
        "mse_raw",
        "mse_intervened",
        "ARG_mse",
        "smape_raw",
        "smape_intervened",
        "ARG_smape",
    ]
    raw_df = results_df[results_df["eval_view_name"] == "raw"].copy()
    int_df = results_df[results_df["eval_view_name"] == "intervened"].copy()
    keys = ["dataset_name", "backbone", "lookback", "horizon", "train_view_name"]
    merged = raw_df[keys + ["mae", "mse", "smape"]].merge(
        int_df[keys + ["mae", "mse", "smape"]],
        on=keys,
        suffixes=("_raw", "_intervened"),
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        {
            "dataset_name": merged["dataset_name"],
            "backbone": merged["backbone"],
            "lookback": merged["lookback"].astype(int),
            "horizon": merged["horizon"],
            "train_view_name": merged["train_view_name"],
            "mae_raw": merged["mae_raw"],
            "mae_intervened": merged["mae_intervened"],
            "ARG_mae": (merged["mae_intervened"] - merged["mae_raw"]) / merged["mae_raw"].clip(lower=1e-8),
            "mse_raw": merged["mse_raw"],
            "mse_intervened": merged["mse_intervened"],
            "ARG_mse": (merged["mse_intervened"] - merged["mse_raw"]) / merged["mse_raw"].clip(lower=1e-8),
            "smape_raw": merged["smape_raw"],
            "smape_intervened": merged["smape_intervened"],
            "ARG_smape": (merged["smape_intervened"] - merged["smape_raw"]) / merged["smape_raw"].clip(lower=1e-8),
        }
    )[columns]


def compute_aif_wgr_table(window_errors_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
    if window_errors_df.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    keys = ["dataset_name", "backbone", "lookback", "horizon", "train_view_name", "eval_view_name"]
    for key, group in window_errors_df.groupby(keys, dropna=False):
        group_mean = group.groupby("group_key", dropna=False)["mae"].mean().sort_values(ascending=False)
        if group_mean.empty:
            continue
        rows.append(
            {
                "dataset_name": key[0],
                "backbone": key[1],
                "lookback": int(key[2]),
                "horizon": int(key[3]),
                "train_view_name": key[4],
                "eval_view_name": key[5],
                "worst_group": str(group_mean.index[0]),
                "mean_error": float(group["mae"].mean()),
                "WGR": float(group_mean.iloc[0]),
                "WGR_gap": float(group_mean.iloc[0] - group["mae"].mean()),
                "n_groups": int(group_mean.shape[0]),
                "n_eval_windows": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def compute_aif_ri_table(results_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset_name",
        "lookback",
        "horizon",
        "train_view_name",
        "reference_eval_view_name",
        "compare_eval_view_name",
        "compare_to",
        "top1_flip",
        "mean_rank_shift",
    ]
    if results_df.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (dataset_name, lookback, horizon, train_view_name), group in results_df.groupby(
        ["dataset_name", "lookback", "horizon", "train_view_name"],
        dropna=False,
    ):
        by_view = {
            str(eval_view_name): frame.sort_values(["mae", "backbone"]).reset_index(drop=True)
            for eval_view_name, frame in group.groupby("eval_view_name", dropna=False)
        }
        if "raw" not in by_view:
            continue
        for compare_view in ["balanced", "clean_like", "intervened"]:
            if compare_view not in by_view:
                continue
            ref_frame = by_view["raw"]
            cmp_frame = by_view[compare_view]
            common = sorted(set(ref_frame["backbone"]) & set(cmp_frame["backbone"]))
            if not common:
                continue
            ref_rank = {backbone: idx + 1 for idx, backbone in enumerate(ref_frame["backbone"].tolist()) if backbone in common}
            cmp_rank = {backbone: idx + 1 for idx, backbone in enumerate(cmp_frame["backbone"].tolist()) if backbone in common}
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "lookback": int(lookback),
                    "horizon": int(horizon),
                    "train_view_name": train_view_name,
                    "reference_eval_view_name": "raw",
                    "compare_eval_view_name": compare_view,
                    "compare_to": f"raw->{compare_view}",
                    "top1_flip": int(ref_frame.iloc[0]["backbone"] != cmp_frame.iloc[0]["backbone"]),
                    "mean_rank_shift": float(sum(abs(ref_rank[bb] - cmp_rank[bb]) for bb in common) / len(common)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


__all__ = [
    "build_group_map",
    "build_loader",
    "build_lookup",
    "compare_against_baseline",
    "compute_aif_arg_table",
    "compute_aif_ri_table",
    "compute_aif_wgr_table",
    "group_tail_mean",
    "load_config",
    "resolve_clean_view_name",
]
