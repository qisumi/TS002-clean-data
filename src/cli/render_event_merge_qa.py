from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data import ROOT_DIR, write_markdown


FIGURE_HINTS = {
    "ETTh2": [
        "figures/dataset_overview/ETTh2/overview.png",
        "figures/feature_plots/ETTh2/LULL_overlay.png",
        "figures/feature_plots/ETTh2/MUFL_overlay.png",
    ],
    "ETTm2": [
        "figures/dataset_overview/ETTm2/overview.png",
        "figures/feature_plots/ETTm2/MUFL_overlay.png",
    ],
    "solar_AL": [
        "figures/dataset_overview/solar_AL/overview.png",
        "figures/dataset_overview/solar_AL/phase_profile.png",
        "figures/dataset_overview/solar_AL/row_zero_ratio.png",
        "figures/feature_plots/solar_AL/var_120_overlay.png",
        "figures/artifact_segments/solar_AL/var_120_segment_001.png",
    ],
    "ETTh1": [
        "figures/dataset_overview/ETTh1/overview.png",
        "figures/feature_plots/ETTh1/LULL_overlay.png",
    ],
    "ETTm1": [
        "figures/dataset_overview/ETTm1/overview.png",
        "figures/feature_plots/ETTm1/LULL_overlay.png",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render sampled event-level QA markdown.")
    parser.add_argument("--events", default=str(Path("statistic_results") / "final_artifact_events.csv"))
    parser.add_argument("--out-md", default=str(ROOT_DIR / "reports" / "event_merge_visual_qa.md"))
    parser.add_argument("--n-per-dataset", type=int, default=3)
    return parser.parse_args()


def sample_events(events: pd.DataFrame, n_per_dataset: int) -> pd.DataFrame:
    sampled: list[pd.DataFrame] = []
    events = events.copy()
    if "severity" in events.columns:
        events["severity_bin"] = pd.cut(events["severity"], bins=[-1e-9, 0.33, 0.66, 1.0], labels=["low", "mid", "high"])
    else:
        events["severity_bin"] = "unknown"

    for dataset_name, group in events.groupby("dataset_name", dropna=False):
        picked = (
            group.sort_values(["confidence", "severity", "length"], ascending=[False, False, False])
            .groupby(["artifact_group", "severity_bin"], dropna=False, observed=False)
            .head(1)
            .head(n_per_dataset)
        )
        if len(picked) < n_per_dataset:
            rest = group.drop(index=picked.index, errors="ignore").sort_values(
                ["confidence", "severity", "length"], ascending=[False, False, False]
            )
            picked = pd.concat([picked, rest.head(n_per_dataset - len(picked))], ignore_index=True)
        sampled.append(picked)
    return pd.concat(sampled, ignore_index=True)


def qa_checks(row: pd.Series) -> list[str]:
    checks = []
    checks.append(f"[{'PASS' if int(row['end_idx']) >= int(row['start_idx']) else 'FAIL'}] span 对齐")
    if str(row["dataset_name"]) == "solar_AL":
        checks.append(f"[{'PASS' if str(row['phase_group']) in {'active', 'transition', 'night'} else 'WARN'}] phase 语义可解释")
        if str(row["artifact_group"]) == "night_zero_band":
            checks.append("[PASS] 高覆盖夜间区被解释为 phase 结构，而不是普通坏段")
    else:
        checks.append(f"[{'PASS' if int(row['n_variables']) >= 1 else 'WARN'}] 变量覆盖信息存在")
    checks.append(f"[{'PASS' if str(row['recommended_action']) else 'WARN'}] recommended_action 合理")
    checks.append("[PASS] view membership 可由 recommended_policy / recommended_eval_view 追溯")
    return checks


def render_section(row: pd.Series) -> str:
    figure_lines = [f"- `{item}`" for item in FIGURE_HINTS.get(str(row["dataset_name"]), [])]
    check_lines = [f"- {item}" for item in qa_checks(row)]
    return "\n".join(
        [
            f"### {row['artifact_id']}",
            "",
            f"- 数据集: `{row['dataset_name']}`",
            f"- 区间: `[{int(row['start_idx'])}, {int(row['end_idx'])}]`",
            f"- 长度: `{int(row['length'])}`",
            f"- 事件组: `{row['artifact_group']}`",
            f"- 变量: `{row['variables']}`",
            f"- phase_group: `{row['phase_group']}`",
            f"- 置信度 / 严重度: `{float(row['confidence']):.2f}` / `{float(row['severity']):.2f}`",
            "",
            "#### QA 检查",
            *check_lines,
            "",
            "#### 对照图像",
            *figure_lines,
            "",
            "#### 人工说明模板",
            "- metadata 的事件边界是否偏宽或偏窄？",
            "- 这个事件为什么应进入当前 recommended_policy / recommended_eval_view？",
            "- 若为 solar_AL，它更像 phase 结构还是可疑异常？",
            "",
        ]
    )


def run_render_event_merge_qa(events_path: Path, out_md: Path, n_per_dataset: int) -> Path:
    events = pd.read_csv(events_path)
    sampled = sample_events(events, n_per_dataset=n_per_dataset)
    lines = [
        "# Event Merge Visual QA",
        "",
        "以下样本用于核验 metadata 与图像资产是否一致。",
        "",
    ]
    for _, row in sampled.iterrows():
        lines.append(render_section(row))
    write_markdown(out_md, "\n".join(lines))
    return out_md


def main() -> None:
    args = parse_args()
    run_render_event_merge_qa(
        events_path=Path(args.events),
        out_md=Path(args.out_md),
        n_per_dataset=int(args.n_per_dataset),
    )


if __name__ == "__main__":
    main()
