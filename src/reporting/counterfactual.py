from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARG_COLUMNS = [
    "scope",
    "pair_name",
    "dataset_name",
    "backbone",
    "lookback",
    "horizon",
    "seed",
    "train_view_token",
    "train_view_name",
    "subset_name",
    "eval_protocol",
    "reference_eval_view_name",
    "compare_eval_view_name",
    "group_key",
    "mae_raw",
    "mae_counterfactual",
    "mae_intervened",
    "ARG_mae",
    "mse_raw",
    "mse_counterfactual",
    "mse_intervened",
    "ARG_mse",
    "smape_raw",
    "smape_counterfactual",
    "smape_intervened",
    "ARG_smape",
]

WGR_COLUMNS = [
    "dataset_name",
    "backbone",
    "lookback",
    "horizon",
    "seed",
    "eval_protocol",
    "subset_name",
    "train_view_name",
    "train_view_token",
    "eval_view_name",
    "eval_view_token",
    "status",
    "skip_reason",
    "worst_group",
    "mean_error",
    "WGR",
    "WGR_gap",
    "n_groups",
    "n_eval_windows",
]

RI_COLUMNS = [
    "dataset_name",
    "lookback",
    "horizon",
    "seed",
    "train_view_token",
    "subset_name",
    "eval_protocol",
    "train_view_name",
    "reference_eval_view_name",
    "compare_eval_view_name",
    "compare_to",
    "top1_flip",
    "mean_rank_shift",
]


def overall_arg_rows(results_df: pd.DataFrame) -> list[dict[str, object]]:
    keys = [
        "dataset_name",
        "backbone",
        "lookback",
        "horizon",
        "seed",
        "train_view_token",
        "train_view_name",
        "subset_name",
        "eval_protocol",
    ]
    completed = results_df[results_df["status"] == "completed"].copy()
    pair_specs = [
        ("standard_intervened", "R", "I"),
        ("paired_input_only", "PR", "PI"),
    ]

    rows: list[dict[str, object]] = []
    for pair_name, raw_token, compare_token in pair_specs:
        raw_df = completed[completed["eval_view_token"] == raw_token][keys + ["eval_view_name", "mae", "mse", "smape"]]
        cmp_df = completed[completed["eval_view_token"] == compare_token][keys + ["eval_view_name", "mae", "mse", "smape"]]
        merged = raw_df.merge(cmp_df, on=keys, suffixes=("_raw", "_counterfactual"))
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "scope": "overall",
                    "pair_name": pair_name,
                    "dataset_name": row.dataset_name,
                    "backbone": row.backbone,
                    "lookback": int(row.lookback),
                    "horizon": int(row.horizon),
                    "seed": int(row.seed),
                    "train_view_token": row.train_view_token,
                    "train_view_name": row.train_view_name,
                    "subset_name": row.subset_name,
                    "eval_protocol": row.eval_protocol,
                    "reference_eval_view_name": row.eval_view_name_raw,
                    "compare_eval_view_name": row.eval_view_name_counterfactual,
                    "group_key": "ALL",
                    "mae_raw": float(row.mae_raw),
                    "mae_counterfactual": float(row.mae_counterfactual),
                    "mae_intervened": float(row.mae_counterfactual),
                    "ARG_mae": float((row.mae_counterfactual - row.mae_raw) / max(row.mae_raw, 1e-8)),
                    "mse_raw": float(row.mse_raw),
                    "mse_counterfactual": float(row.mse_counterfactual),
                    "mse_intervened": float(row.mse_counterfactual),
                    "ARG_mse": float((row.mse_counterfactual - row.mse_raw) / max(row.mse_raw, 1e-8)),
                    "smape_raw": float(row.smape_raw),
                    "smape_counterfactual": float(row.smape_counterfactual),
                    "smape_intervened": float(row.smape_counterfactual),
                    "ARG_smape": float((row.smape_counterfactual - row.smape_raw) / max(row.smape_raw, 1e-8)),
                }
            )
    return rows


def group_arg_rows(window_errors_df: pd.DataFrame) -> list[dict[str, object]]:
    keys = [
        "dataset_name",
        "backbone",
        "lookback",
        "horizon",
        "seed",
        "train_view_token",
        "train_view_name",
        "subset_name",
        "eval_protocol",
        "group_key",
    ]
    grouped = (
        window_errors_df.groupby(keys + ["eval_view_token"], dropna=False)[["mae", "mse", "smape"]]
        .mean()
        .reset_index()
    )

    rows: list[dict[str, object]] = []
    for pair_name, raw_token, compare_token in [("standard_intervened", "R", "I"), ("paired_input_only", "PR", "PI")]:
        raw_df = grouped[grouped["eval_view_token"] == raw_token].drop(columns=["eval_view_token"])
        cmp_df = grouped[grouped["eval_view_token"] == compare_token].drop(columns=["eval_view_token"])
        merged = raw_df.merge(cmp_df, on=keys, suffixes=("_raw", "_counterfactual"))
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "scope": "group",
                    "pair_name": pair_name,
                    "dataset_name": row.dataset_name,
                    "backbone": row.backbone,
                    "lookback": int(row.lookback),
                    "horizon": int(row.horizon),
                    "seed": int(row.seed),
                    "train_view_token": row.train_view_token,
                    "train_view_name": row.train_view_name,
                    "subset_name": row.subset_name,
                    "eval_protocol": row.eval_protocol,
                    "group_key": row.group_key,
                    "mae_raw": float(row.mae_raw),
                    "mae_counterfactual": float(row.mae_counterfactual),
                    "mae_intervened": float(row.mae_counterfactual),
                    "ARG_mae": float((row.mae_counterfactual - row.mae_raw) / max(row.mae_raw, 1e-8)),
                    "mse_raw": float(row.mse_raw),
                    "mse_counterfactual": float(row.mse_counterfactual),
                    "mse_intervened": float(row.mse_counterfactual),
                    "ARG_mse": float((row.mse_counterfactual - row.mse_raw) / max(row.mse_raw, 1e-8)),
                    "smape_raw": float(row.smape_raw),
                    "smape_counterfactual": float(row.smape_counterfactual),
                    "smape_intervened": float(row.smape_counterfactual),
                    "ARG_smape": float((row.smape_counterfactual - row.smape_raw) / max(row.smape_raw, 1e-8)),
                }
            )
    return rows


def compute_wgr_table(window_errors_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    setting_cols = [
        "dataset_name",
        "backbone",
        "lookback",
        "horizon",
        "seed",
        "eval_protocol",
        "subset_name",
        "train_view_name",
        "train_view_token",
        "eval_view_name",
        "eval_view_token",
    ]
    for key, group in window_errors_df.groupby(setting_cols, dropna=False):
        group_mean = group.groupby("group_key", dropna=False)["mae"].mean().sort_values(ascending=False)
        if group_mean.empty:
            continue
        overall = float(group["mae"].mean())
        worst_group = str(group_mean.index[0])
        worst_value = float(group_mean.iloc[0])
        rows.append(
            {
                "dataset_name": key[0],
                "backbone": key[1],
                "lookback": int(key[2]),
                "horizon": int(key[3]),
                "seed": int(key[4]),
                "eval_protocol": key[5],
                "subset_name": key[6],
                "train_view_name": key[7],
                "train_view_token": key[8],
                "eval_view_name": key[9],
                "eval_view_token": key[10],
                "status": "completed",
                "skip_reason": "",
                "worst_group": worst_group,
                "mean_error": overall,
                "WGR": worst_value,
                "WGR_gap": float(worst_value - overall),
                "n_groups": int(group_mean.shape[0]),
                "n_eval_windows": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=WGR_COLUMNS)


def compute_ri_table(results_df: pd.DataFrame) -> pd.DataFrame:
    completed = results_df[results_df["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame(columns=RI_COLUMNS)

    rows: list[dict[str, object]] = []
    pairs = [("R", "C"), ("R", "I"), ("PR", "PI")]
    grouping_cols = ["dataset_name", "lookback", "horizon", "seed", "train_view_token", "subset_name", "eval_protocol"]

    for key, group in completed.groupby(grouping_cols, dropna=False):
        by_eval = {token: frame.sort_values(["mae", "backbone"]).reset_index(drop=True) for token, frame in group.groupby("eval_view_token")}
        for ref_token, cmp_token in pairs:
            if ref_token not in by_eval or cmp_token not in by_eval:
                continue
            ref_frame = by_eval[ref_token]
            cmp_frame = by_eval[cmp_token]
            common = sorted(set(ref_frame["backbone"]) & set(cmp_frame["backbone"]))
            if not common:
                continue
            ref_rank = {backbone: rank + 1 for rank, backbone in enumerate(ref_frame["backbone"].tolist()) if backbone in common}
            cmp_rank = {backbone: rank + 1 for rank, backbone in enumerate(cmp_frame["backbone"].tolist()) if backbone in common}
            top1_flip = int(ref_frame.iloc[0]["backbone"] != cmp_frame.iloc[0]["backbone"])
            mean_rank_shift = float(np.mean([abs(ref_rank[bb] - cmp_rank[bb]) for bb in common]))
            rows.append(
                {
                    "dataset_name": key[0],
                    "lookback": int(key[1]),
                    "horizon": int(key[2]),
                    "seed": int(key[3]),
                    "train_view_token": key[4],
                    "subset_name": key[5],
                    "eval_protocol": key[6],
                    "train_view_name": str(group["train_view_name"].iloc[0]),
                    "reference_eval_view_name": str(ref_frame.iloc[0]["eval_view_name"]),
                    "compare_eval_view_name": str(cmp_frame.iloc[0]["eval_view_name"]),
                    "compare_to": f"{ref_token}->{cmp_token}",
                    "top1_flip": top1_flip,
                    "mean_rank_shift": mean_rank_shift,
                }
            )
    return pd.DataFrame(rows, columns=RI_COLUMNS)


def build_summary_markdown(
    backbone_status: dict[str, dict[str, str]],
    results_df: pd.DataFrame,
    arg_df: pd.DataFrame,
    wgr_df: pd.DataFrame,
    ri_df: pd.DataFrame,
    setting_logs_dir: Path,
) -> str:
    lines = [
        "# Counterfactual Eval Summary",
        "",
        "本轮已将 006 从 setting matrix 占位执行，升级为真实的 `DLinear / PatchTST / TQNet / iTransformer / ModernTCN / TimeMixer / TimeMixerPP` 训练与窗口级误差评测。",
        "",
        "## Backbone 状态",
        "",
    ]
    for backbone, info in backbone_status.items():
        suffix = f" (`{info['repo_path']}`)" if info["repo_path"] else ""
        lines.append(f"- {backbone}: {info['status']}{suffix}")

    status_counts = dict(results_df["status"].value_counts()) if not results_df.empty and "status" in results_df.columns else {}
    lines.extend(
        [
            "",
            "## 结果状态",
            "",
            f"- setting 数: {len(results_df)}",
            f"- status 分布: {status_counts}",
        ]
    )

    completed = results_df[results_df["status"] == "completed"].copy() if "status" in results_df.columns else pd.DataFrame()
    if completed.empty:
        lines.extend(["- 当前没有 completed setting。", ""])
        return "\n".join(lines)

    lines.extend(
        [
            f"- completed setting 数: {len(completed)}",
            f"- 平均 MAE: {completed['mae'].mean():.4f}",
            "",
            "## 主要发现",
            "",
        ]
    )

    overall_arg = (
        arg_df[arg_df["scope"] == "overall"].sort_values("ARG_mae", ascending=False)
        if "scope" in arg_df.columns
        else pd.DataFrame(columns=ARG_COLUMNS)
    )
    if not overall_arg.empty:
        top_arg = overall_arg.head(5)
        for row in top_arg.itertuples(index=False):
            lines.append(
                f"- ARG↑ {row.dataset_name} / {row.backbone} / L{int(row.lookback)} / H{row.horizon} / train={row.train_view_name} / "
                f"{row.pair_name} / subset={row.subset_name}: {row.ARG_mae:.4f}"
            )
    else:
        lines.append("- 当前没有可计算的 ARG pair；本轮 completed setting 未形成 `raw vs counterfactual` 成对结果。")

    if not wgr_df.empty:
        top_wgr = wgr_df.sort_values("WGR_gap", ascending=False).head(5)
        for row in top_wgr.itertuples(index=False):
            lines.append(
                f"- WGR_gap↑ {row.dataset_name} / {row.backbone} / L{int(row.lookback)} / H{row.horizon} / {row.train_view_name}->{row.eval_view_name} / "
                f"subset={row.subset_name}: "
                f"{row.WGR_gap:.4f} (worst_group={row.worst_group})"
            )

    if not ri_df.empty:
        flip_count = int(ri_df["top1_flip"].sum())
        lines.append(f"- 排名翻转次数: {flip_count}")
        top_shift = ri_df.sort_values("mean_rank_shift", ascending=False).head(5)
        for row in top_shift.itertuples(index=False):
            lines.append(
                f"- RI {row.dataset_name} / L{int(row.lookback)} / H{row.horizon} / train={row.train_view_name} / "
                f"{row.compare_to} / subset={row.subset_name}: "
                f"top1_flip={row.top1_flip}, mean_rank_shift={row.mean_rank_shift:.4f}"
            )

    lines.extend(
        [
            "",
            "## 产物",
            "",
            "- `counterfactual_2x2.csv`: 聚合 setting 级结果",
            "- `counterfactual_window_errors.csv`: 窗口级误差长表",
            "- `artifact_reliance_gap.csv`: `ARG` 总表",
            "- `worst_group_risk.csv`: `WGR` 总表",
            "- `ranking_instability.csv`: `RI` 总表",
            f"- `{setting_logs_dir}`: 每个 setting 的单独日志",
            "",
        ]
    )
    return "\n".join(lines)
