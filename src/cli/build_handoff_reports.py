from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from data import ROOT_DIR, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 011 handoff-facing reports from standard pipeline outputs.")
    parser.add_argument("--results-dir", default=str(Path("results")))
    parser.add_argument("--reports-dir", default=str(Path("reports")))
    parser.add_argument("--stats-dir", default=str(Path("statistic_results")))
    return parser.parse_args()


def read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def count_rows_for_dataset(df: pd.DataFrame, dataset_name: str) -> int:
    if df.empty or "dataset_name" not in df.columns:
        return 0
    return int(df["dataset_name"].astype(str).eq(dataset_name).sum())


def fmt_lb_h(row: pd.Series | Any) -> str:
    lookback = int(row.lookback) if hasattr(row, "lookback") and not pd.isna(row.lookback) else None
    horizon = int(row.horizon) if hasattr(row, "horizon") and not pd.isna(row.horizon) else None
    if lookback is None:
        return f"H{horizon}" if horizon is not None else "NA"
    if horizon is None:
        return f"L{lookback}"
    return f"L{lookback} / H{horizon}"


def completed_counterfactual_rows(counterfactual_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if counterfactual_df.empty:
        return pd.DataFrame()
    mask = (
        counterfactual_df["status"].astype(str).eq("completed")
        & counterfactual_df["dataset_name"].astype(str).eq(dataset_name)
    )
    return counterfactual_df.loc[mask].copy()


def best_raw_rows(counterfactual_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    subset = completed_counterfactual_rows(counterfactual_df, dataset_name)
    if subset.empty:
        return subset
    subset = subset[
        subset["eval_protocol"].astype(str).eq("view_matrix")
        & subset["train_view_name"].astype(str).eq("raw")
        & subset["eval_view_name"].astype(str).eq("raw")
    ].copy()
    if subset.empty:
        return subset
    sort_cols = ["horizon", "mae", "backbone"]
    if "lookback" in subset.columns:
        sort_cols.append("lookback")
    return (
        subset.sort_values(sort_cols)
        .groupby("horizon", dropna=False)
        .head(3)
        .reset_index(drop=True)
    )


def overall_arg_rows(arg_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if arg_df.empty:
        return pd.DataFrame()
    out = arg_df[
        arg_df["dataset_name"].astype(str).eq(dataset_name)
        & arg_df["scope"].astype(str).eq("overall")
    ].copy()
    if out.empty:
        return out
    return out.sort_values("ARG_mae", ascending=False).reset_index(drop=True)


def merge_with_best_baseline(model_df: pd.DataFrame, baseline_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if model_df.empty or baseline_df.empty:
        return pd.DataFrame()
    model_subset = model_df[model_df["dataset_name"].astype(str).eq(dataset_name)].copy()
    baseline = baseline_df[
        baseline_df["status"].astype(str).eq("completed")
        & baseline_df["dataset_name"].astype(str).eq(dataset_name)
        & baseline_df["train_view_name"].astype(str).eq("raw")
    ].copy()
    if model_subset.empty or baseline.empty:
        return pd.DataFrame()
    merge_cols = ["dataset_name", "horizon", "eval_view_name"]
    if "lookback" in model_subset.columns and "lookback" in baseline.columns:
        merge_cols.insert(1, "lookback")
    baseline = (
        baseline.sort_values(merge_cols + ["mae", "backbone"])
        .groupby(merge_cols, dropna=False)
        .head(1)
        .rename(columns={"mae": "baseline_mae", "backbone": "baseline_backbone"})
        .reset_index(drop=True)
    )
    keep_cols = merge_cols + ["baseline_backbone", "baseline_mae"]
    merged = model_subset.merge(baseline[keep_cols], on=merge_cols, how="left")
    if merged.empty:
        return merged
    merged["delta_vs_baseline"] = merged["mae"] - merged["baseline_mae"]
    return merged


def build_main_report(
    counterfactual_df: pd.DataFrame,
    arg_df: pd.DataFrame,
    wgr_df: pd.DataFrame,
    ri_df: pd.DataFrame,
    etth2_support_df: pd.DataFrame,
    etth2_var_df: pd.DataFrame,
    aef_df: pd.DataFrame,
    aef_control_df: pd.DataFrame,
    aef_plus_df: pd.DataFrame,
    aif_plus_df: pd.DataFrame,
    aif_plus_arg_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> str:
    lines = [
        "# 011 Next Step Results Report",
        "",
        "这份报告按完整 pipeline 重写 011 汇总：先回看前序阶段是否闭环，再收敛到 `ETTh2 / solar_AL / ETTh1` 的下一步判断。",
        "",
    ]

    completed = counterfactual_df[counterfactual_df["status"].astype(str) == "completed"].copy() if not counterfactual_df.empty else pd.DataFrame()
    if not completed.empty:
        dataset_counts = completed.groupby("dataset_name", dropna=False).size().to_dict()
        lines.extend(
            [
                "## Coverage",
                "",
                f"- completed setting 数: {len(completed)}",
                f"- 数据集覆盖: {dataset_counts}",
                f"- backbone 覆盖: {sorted(completed['backbone'].astype(str).unique().tolist())}",
                "",
            ]
        )
    else:
        lines.extend(["## Coverage", "", "- 目前还没有 completed counterfactual setting。", ""])

    lines.extend(["## Earlier Stages", ""])
    stage0_datasets = sorted(set(completed["dataset_name"].astype(str).tolist())) if not completed.empty else []
    lines.append(f"- 统计/检测阶段覆盖数据集: {stage0_datasets if stage0_datasets else '尚未产出'}")
    if not etth2_support_df.empty:
        lines.append(f"- 004/005/011 侧的 ETTh2 channel support 行数: {len(etth2_support_df)}")
    if not arg_df.empty:
        lines.append(f"- 006 counterfactual ARG 行数: {len(arg_df)}")
    if not aef_df.empty:
        lines.append(f"- 007 AEF-Weak 行数: {len(aef_df)}")
    if not aef_control_df.empty:
        lines.append(f"- 007 AEF-Control 行数: {len(aef_control_df)}")
    if not aef_plus_df.empty:
        lines.append(f"- 008 AEF-Plus 行数: {len(aef_plus_df)}")
    if not aif_plus_df.empty:
        lines.append(f"- 010 AIF-Plus 行数: {len(aif_plus_df)}")
    lines.extend(
        [
            "- 前序阶段对应文件仍保留在标准目录，并会在 handoff 时进入 `archive_not_for_handoff/`。",
            "",
        ]
    )

    lines.extend(["## 004 Event Merge / 005 Views", ""])
    lines.append("- `artifact_event_summary.md` 提供 event 级聚合判断，`event_merge_visual_qa.md` 保留抽样核查。")
    lines.append("- `eval_view_design.md` 和 `clean_view_qc_report.md` 负责把 event 映射成 window-level view，并检查 support/fallback。")
    if not etth2_support_df.empty:
        support_status = etth2_support_df["status"].astype(str).value_counts().to_dict()
        lines.append(f"- ETTh2 channel support 状态分布: {support_status}")
    lines.append("")

    lines.extend(["## 006 Counterfactual", ""])
    if not completed.empty:
        for row in completed.sort_values(["mae", "dataset_name"]).head(6).itertuples(index=False):
            lines.append(
                f"- {row.dataset_name} / {row.backbone} / {fmt_lb_h(row)} / train={row.train_view_name} / eval={row.eval_view_name}: "
                f"MAE={float(row.mae):.4f}"
            )
    else:
        lines.append("- 尚无 006 completed 结果。")
    lines.append("")

    lines.extend(["## ETTh2", ""])
    if not etth2_support_df.empty:
        for row in etth2_support_df.sort_values(["lookback", "horizon", "status", "channel"]).head(10).itertuples(index=False):
            lines.append(
                f"- {fmt_lb_h(row)} / {row.channel}: status={row.status}, "
                f"target_clean_support={int(row.target_clean_support)}, input_flag_rate={float(row.input_flag_rate):.3f}"
            )
    if not etth2_var_df.empty:
        var_rows = etth2_var_df.copy()
        var_rows["delta_mae"] = var_rows["mae_input_intervened"] - var_rows["mae_raw"]
        for row in var_rows.sort_values("delta_mae").head(6).itertuples(index=False):
            lines.append(
                f"- variable-stratified {fmt_lb_h(row)} / {row.backbone} / train={row.train_view_name} / "
                f"{row.channel_subset}: raw={row.mae_raw:.4f}, input_only={row.mae_input_intervened:.4f}, delta={row.delta_mae:.4f}"
            )
    if etth2_support_df.empty and etth2_var_df.empty:
        lines.append("- ETTh2 诊断结果尚未生成。")

    lines.extend(["", "## solar_AL", ""])
    solar_arg = overall_arg_rows(arg_df, "solar_AL")
    if not solar_arg.empty:
        for row in solar_arg.head(6).itertuples(index=False):
            lines.append(
                f"- {fmt_lb_h(row)} / {row.backbone} / train={row.train_view_name} / {row.pair_name}: ARG={row.ARG_mae:.4f}"
            )
    else:
        lines.append("- 尚无 solar_AL 的 ARG 汇总。")

    solar_aif_plus = merge_with_best_baseline(aif_plus_df, baseline_df, "solar_AL")
    if not solar_aif_plus.empty:
        for row in solar_aif_plus.sort_values("delta_vs_baseline").head(4).itertuples(index=False):
            lines.append(
                f"- AIF-Plus {fmt_lb_h(row)} / {row.eval_view_name}: "
                f"AIF={row.mae:.4f}, best_ERM={row.baseline_mae:.4f} ({row.baseline_backbone}), delta={row.delta_vs_baseline:.4f}"
            )

    lines.extend(["", "## ETTh1", ""])
    etth1_best = best_raw_rows(counterfactual_df, "ETTh1")
    if not etth1_best.empty:
        for row in etth1_best.itertuples(index=False):
            lines.append(f"- {fmt_lb_h(row)} / {row.backbone} / raw->raw: MAE={row.mae:.4f}")
    etth1_ri = ri_df[ri_df["dataset_name"].astype(str).eq("ETTh1")].copy() if not ri_df.empty else pd.DataFrame()
    if not etth1_ri.empty:
        for row in etth1_ri.sort_values("mean_rank_shift", ascending=False).head(4).itertuples(index=False):
            lines.append(
                f"- RI {fmt_lb_h(row)} / {row.compare_to}: top1_flip={int(row.top1_flip)}, mean_rank_shift={float(row.mean_rank_shift):.4f}"
            )
    if etth1_best.empty and etth1_ri.empty:
        lines.append("- 尚无 ETTh1 的 clean-ish 对照结果。")

    lines.extend(["", "## AEF / Plus Models", ""])
    if not aef_df.empty and not aef_control_df.empty:
        aef_merge_cols = ["dataset_name", "horizon", "eval_view_name"]
        if "lookback" in aef_df.columns and "lookback" in aef_control_df.columns:
            aef_merge_cols.insert(1, "lookback")
        merged = aef_df.merge(
            aef_control_df,
            on=aef_merge_cols,
            suffixes=("_weak", "_control"),
        )
        if not merged.empty:
            merged["shortcut_gain"] = merged["mae_control"] - merged["mae_weak"]
            for row in merged[merged["eval_view_name"] == "raw"].sort_values("shortcut_gain", ascending=False).head(6).itertuples(index=False):
                lines.append(
                    f"- AEF raw {fmt_lb_h(row)} / {row.dataset_name}: "
                    f"Weak={row.mae_weak:.4f}, Control={row.mae_control:.4f}, shortcut_gain={row.shortcut_gain:.4f}"
                )
    if not aef_plus_df.empty:
        for row in aef_plus_df.sort_values("mae").head(4).itertuples(index=False):
            lines.append(
                f"- AEF-Plus {fmt_lb_h(row)} / {row.dataset_name} / {row.eval_view_name}: MAE={row.mae:.4f}"
            )
    if not aif_plus_arg_df.empty:
        for row in aif_plus_arg_df.sort_values("ARG_mae", ascending=False).head(6).itertuples(index=False):
            lines.append(
                f"- AIF-Plus ARG {fmt_lb_h(row)} / {row.dataset_name}: {row.ARG_mae:.4f}"
            )

    lines.extend(
        [
            "",
            "## Add-on Datasets",
            "",
        ]
    )
    for dataset_name in ["weather", "exchange_rate", "electricity", "ETTm1", "ETTm2"]:
        addon_best = best_raw_rows(counterfactual_df, dataset_name)
        if addon_best.empty:
            continue
        row = addon_best.iloc[0]
        lookback_text = f"L{int(row['lookback'])}/" if "lookback" in addon_best.columns else ""
        lines.append(
            f"- {dataset_name}: best raw baseline is {row['backbone']} at {lookback_text}H{int(row['horizon'])} "
            f"with MAE={float(row['mae']):.4f}"
        )

    lines.extend(
        [
            "",
            "## Linked Pages",
            "",
            "- `reports/artifact_event_summary.md`",
            "- `reports/event_merge_visual_qa.md`",
            "- `reports/eval_view_design.md`",
            "- `reports/counterfactual_eval_summary.md`",
            "- `reports/aef_summary.md`",
            "- `reports/aef_plus_summary.md`",
            "- `reports/aif_plus_summary.md`",
            "- `reports/unified_leaderboard_appendix.md`",
            "- `reports/011_ETTh1_results_page.md`",
            "- `reports/011_weather_results_page.md`",
            "- `reports/011_support_bundle_manifest.md`",
            "",
        ]
    )
    return "\n".join(lines)


def build_dataset_page(
    dataset_name: str,
    title: str,
    counterfactual_df: pd.DataFrame,
    arg_df: pd.DataFrame,
    wgr_df: pd.DataFrame,
    ri_df: pd.DataFrame,
    aif_plus_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> str:
    lines = [f"# {title}", "", f"这页只保留 `{dataset_name}` 的结果，便于单独查 clean-ish 对照与附录结论。", ""]

    best_rows = best_raw_rows(counterfactual_df, dataset_name)
    lines.extend(["## Raw Leaderboard", ""])
    if best_rows.empty:
        lines.append("- 尚无 completed raw baseline。")
    else:
        for row in best_rows.itertuples(index=False):
            lines.append(f"- {fmt_lb_h(row)} / {row.backbone}: MAE={row.mae:.4f}, MSE={row.mse:.4f}, sMAPE={row.smape:.4f}")

    ds_arg = overall_arg_rows(arg_df, dataset_name)
    lines.extend(["", "## ARG / WGR / RI", ""])
    if not ds_arg.empty:
        for row in ds_arg.head(6).itertuples(index=False):
            lines.append(f"- ARG {fmt_lb_h(row)} / {row.backbone} / {row.pair_name}: {row.ARG_mae:.4f}")
    ds_wgr = wgr_df[wgr_df["dataset_name"].astype(str).eq(dataset_name)].copy() if not wgr_df.empty else pd.DataFrame()
    if not ds_wgr.empty:
        for row in ds_wgr.sort_values("WGR_gap", ascending=False).head(4).itertuples(index=False):
            lines.append(
                f"- WGR {fmt_lb_h(row)} / {row.backbone} / {row.eval_view_name}: {row.WGR_gap:.4f} (worst_group={row.worst_group})"
            )
    ds_ri = ri_df[ri_df["dataset_name"].astype(str).eq(dataset_name)].copy() if not ri_df.empty else pd.DataFrame()
    if not ds_ri.empty:
        for row in ds_ri.sort_values("mean_rank_shift", ascending=False).head(4).itertuples(index=False):
            lines.append(
                f"- RI {fmt_lb_h(row)} / {row.compare_to}: top1_flip={int(row.top1_flip)}, mean_rank_shift={float(row.mean_rank_shift):.4f}"
            )
    if ds_arg.empty and ds_wgr.empty and ds_ri.empty:
        lines.append("- 尚无 robustness 汇总。")

    lines.extend(["", "## AIF-Plus vs Best ERM", ""])
    merged_aif = merge_with_best_baseline(aif_plus_df, baseline_df, dataset_name)
    if merged_aif.empty:
        lines.append("- 尚无 AIF-Plus 对齐结果。")
    else:
        for row in merged_aif.sort_values("delta_vs_baseline").head(6).itertuples(index=False):
            lines.append(
                f"- {fmt_lb_h(row)} / {row.eval_view_name}: "
                f"AIF={row.mae:.4f}, best_ERM={row.baseline_mae:.4f} ({row.baseline_backbone}), delta={row.delta_vs_baseline:.4f}"
            )

    lines.append("")
    return "\n".join(lines)


def build_support_manifest(root_dir: Path) -> str:
    key_files = [
        "reports/artifact_event_summary.md",
        "reports/event_merge_visual_qa.md",
        "reports/eval_view_design.md",
        "reports/counterfactual_eval_summary.md",
        "reports/aef_summary.md",
        "reports/aef_plus_summary.md",
        "reports/aif_plus_summary.md",
        "reports/unified_leaderboard_appendix.md",
        "reports/011_next_step_results_report.md",
        "reports/011_step0_baseline_diagnosis.md",
        "reports/clean_view_qc_report.md",
        "reports/clean_view_support_summary.csv",
        "reports/011_ETTh1_results_page.md",
        "reports/011_weather_results_page.md",
        "results/counterfactual_2x2.csv",
        "results/artifact_reliance_gap.csv",
        "results/worst_group_risk.csv",
        "results/ranking_instability.csv",
        "results/etth2_channel_support.csv",
        "results/etth2_variable_stratified_eval.csv",
        "results/aef_results.csv",
        "results/aef_control_results.csv",
        "results/aef_plus_results.csv",
        "results/aef_plus_artifact_reliance_gap.csv",
        "results/aef_plus_worst_group_risk.csv",
        "results/aif_plus_results.csv",
        "results/aif_plus_artifact_reliance_gap.csv",
        "results/aif_plus_worst_group_risk.csv",
        "results/aif_plus_ranking_instability.csv",
        "results/unified_leaderboard_rows.csv",
        "results/unified_leaderboard_summary.csv",
        "results/unified_significance.csv",
        "results/unified_error_distribution.csv",
        "statistic_results/dataset_registry.csv",
        "statistic_results/final_artifact_events.csv",
        "statistic_results/final_artifact_metadata.csv",
        "statistic_results/solar_AL_phase_annotations.csv",
        "statistic_results/solar_AL_phase_profile.csv",
    ]
    lines = [
        "# 011 Support Bundle Manifest",
        "",
        "下表记录 handoff 相关主文件是否存在，以及 CSV 的行数概览。",
        "",
        "| path | status | rows |",
        "| --- | --- | ---: |",
    ]
    for rel_path in key_files:
        path = root_dir / rel_path
        if not path.exists():
            lines.append(f"| `{rel_path}` | missing | 0 |")
            continue
        if path.suffix.lower() == ".csv":
            rows = len(pd.read_csv(path, low_memory=False))
            lines.append(f"| `{rel_path}` | present | {rows} |")
        else:
            lines.append(f"| `{rel_path}` | present | 1 |")
    lines.append("")
    return "\n".join(lines)


def build_addon_dataset_status_summary(
    support_df: pd.DataFrame,
    events_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    arg_df: pd.DataFrame,
    wgr_df: pd.DataFrame,
    ri_df: pd.DataFrame,
    aef_df: pd.DataFrame,
    aef_control_df: pd.DataFrame,
    aef_plus_df: pd.DataFrame,
    aif_plus_df: pd.DataFrame,
    aif_plus_arg_df: pd.DataFrame,
    aif_plus_wgr_df: pd.DataFrame,
    aif_plus_ri_df: pd.DataFrame,
) -> str:
    addon_datasets = ["ETTm1", "ETTm2", "weather", "exchange_rate", "electricity"]
    lines = [
        "# Add-on Dataset Status Summary",
        "",
        "这份说明专门汇总 `ETTm1 / ETTm2 / weather / exchange_rate / electricity` 的当前状态。",
        "重点区分两类信息：",
        "- 上游证据是否已经生成，例如 event merge、view design、clean/intervened support。",
        "- 最终实验结果是否真正落到了 `results/*.csv` 聚合表。",
        "",
    ]

    result_tables = [
        ("counterfactual_2x2", counterfactual_df),
        ("artifact_reliance_gap", arg_df),
        ("worst_group_risk", wgr_df),
        ("ranking_instability", ri_df),
        ("aef_results", aef_df),
        ("aef_control_results", aef_control_df),
        ("aef_plus_results", aef_plus_df),
        ("aif_plus_results", aif_plus_df),
        ("aif_plus_artifact_reliance_gap", aif_plus_arg_df),
        ("aif_plus_worst_group_risk", aif_plus_wgr_df),
        ("aif_plus_ranking_instability", aif_plus_ri_df),
    ]

    for dataset_name in addon_datasets:
        lines.extend([f"## {dataset_name}", ""])
        support_rows = support_df[support_df["dataset_name"].astype(str).eq(dataset_name)].copy() if not support_df.empty else pd.DataFrame()
        event_rows = events_df[events_df["dataset_name"].astype(str).eq(dataset_name)].copy() if not events_df.empty else pd.DataFrame()

        if support_rows.empty:
            lines.append("- view/QC: missing")
        else:
            lookbacks = sorted(support_rows["lookback"].dropna().astype(int).unique().tolist())
            horizons = sorted(support_rows["horizon"].dropna().astype(int).unique().tolist())
            view_status = support_rows["view_status"].astype(str).value_counts().to_dict()
            lines.append(f"- view/QC lookbacks: {lookbacks}")
            lines.append(f"- view/QC horizons: {horizons}")
            lines.append(f"- view_status: {view_status}")

        if event_rows.empty:
            lines.append("- event merge: missing")
        else:
            lines.append(f"- final_artifact_events rows: {len(event_rows)}")
            if "artifact_group" in event_rows.columns:
                top_groups = event_rows["artifact_group"].astype(str).value_counts().head(3).to_dict()
                lines.append(f"- top artifact groups: {top_groups}")

        lines.append("- final experiment aggregate rows:")
        total_rows = 0
        for table_name, df in result_tables:
            n_rows = count_rows_for_dataset(df, dataset_name)
            total_rows += n_rows
            lines.append(f"  - {table_name}: {n_rows}")
        if total_rows == 0:
            lines.append("- interpretation: 当前标准结果中没有该数据集的最终实验结果行；已有内容只覆盖到上游 event/view/QC 证据。")
        else:
            lines.append("- interpretation: 该数据集已有最终实验结果，见上述聚合表。")
        lines.append("")

    lines.extend(
        [
            "## Reading Note",
            "",
            "- 如果你只想判断这些 add-on 数据集是否“做过前处理闭环”，看 `clean_view_qc_report.md` 和 `clean_view_support_summary.csv`。",
            "- 如果你要判断它们是否“已经完成最终实验”，必须以 `results/*.csv` 中是否出现对应 `dataset_name` 行为准。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root_dir = ROOT_DIR
    results_dir = root_dir / Path(args.results_dir)
    reports_dir = root_dir / Path(args.reports_dir)
    stats_dir = root_dir / Path(args.stats_dir)

    counterfactual_df = read_optional_csv(results_dir / "counterfactual_2x2.csv")
    arg_df = read_optional_csv(results_dir / "artifact_reliance_gap.csv")
    wgr_df = read_optional_csv(results_dir / "worst_group_risk.csv")
    ri_df = read_optional_csv(results_dir / "ranking_instability.csv")
    etth2_support_df = read_optional_csv(results_dir / "etth2_channel_support.csv")
    etth2_var_df = read_optional_csv(results_dir / "etth2_variable_stratified_eval.csv")
    aef_df = read_optional_csv(results_dir / "aef_results.csv")
    aef_control_df = read_optional_csv(results_dir / "aef_control_results.csv")
    aef_plus_df = read_optional_csv(results_dir / "aef_plus_results.csv")
    aif_plus_df = read_optional_csv(results_dir / "aif_plus_results.csv")
    aif_plus_arg_df = read_optional_csv(results_dir / "aif_plus_artifact_reliance_gap.csv")
    aif_plus_wgr_df = read_optional_csv(results_dir / "aif_plus_worst_group_risk.csv")
    aif_plus_ri_df = read_optional_csv(results_dir / "aif_plus_ranking_instability.csv")
    support_df = read_optional_csv(reports_dir / "clean_view_support_summary.csv")
    events_df = read_optional_csv(stats_dir / "final_artifact_events.csv")

    write_markdown(
        reports_dir / "011_next_step_results_report.md",
        build_main_report(
            counterfactual_df=counterfactual_df,
            arg_df=arg_df,
            wgr_df=wgr_df,
            ri_df=ri_df,
            etth2_support_df=etth2_support_df,
            etth2_var_df=etth2_var_df,
            aef_df=aef_df,
            aef_control_df=aef_control_df,
            aef_plus_df=aef_plus_df,
            aif_plus_df=aif_plus_df,
            aif_plus_arg_df=aif_plus_arg_df,
            baseline_df=counterfactual_df,
        ),
    )
    write_markdown(
        reports_dir / "011_ETTh1_results_page.md",
        build_dataset_page(
            dataset_name="ETTh1",
            title="011 ETTh1 Results Page",
            counterfactual_df=counterfactual_df,
            arg_df=arg_df,
            wgr_df=wgr_df,
            ri_df=ri_df,
            aif_plus_df=aif_plus_df,
            baseline_df=counterfactual_df,
        ),
    )
    write_markdown(
        reports_dir / "011_weather_results_page.md",
        build_dataset_page(
            dataset_name="weather",
            title="011 Weather Results Page",
            counterfactual_df=counterfactual_df,
            arg_df=arg_df,
            wgr_df=wgr_df,
            ri_df=ri_df,
            aif_plus_df=aif_plus_df,
            baseline_df=counterfactual_df,
        ),
    )
    write_markdown(
        reports_dir / "addon_dataset_status_summary.md",
        build_addon_dataset_status_summary(
            support_df=support_df,
            events_df=events_df,
            counterfactual_df=counterfactual_df,
            arg_df=arg_df,
            wgr_df=wgr_df,
            ri_df=ri_df,
            aef_df=aef_df,
            aef_control_df=aef_control_df,
            aef_plus_df=aef_plus_df,
            aif_plus_df=aif_plus_df,
            aif_plus_arg_df=aif_plus_arg_df,
            aif_plus_wgr_df=aif_plus_wgr_df,
            aif_plus_ri_df=aif_plus_ri_df,
        ),
    )
    write_markdown(reports_dir / "011_support_bundle_manifest.md", build_support_manifest(root_dir))


if __name__ == "__main__":
    main()
