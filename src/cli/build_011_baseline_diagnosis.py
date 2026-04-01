from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

from data import ROOT_DIR, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 0 baseline diagnosis for 011 next-step execution.")
    parser.add_argument("--views-dir", default=str(Path("statistic_results") / "window_views"))
    parser.add_argument("--manifest", default=str(Path("statistic_results") / "eval_view_manifest.csv"))
    parser.add_argument("--results-dir", default=str(Path("results")))
    parser.add_argument("--out", default=str(Path("reports") / "011_step0_baseline_diagnosis.md"))
    parser.add_argument("--lookbacks", nargs="+", type=int, default=None)
    return parser.parse_args()


VIEW_FILE_RE = re.compile(r"^ETTh2_L(?P<lookback>\d+)_H(?P<horizon>\d+)\.csv$")


def etth2_strict_support_lines(views_dir: Path, lookbacks: list[int] | None = None) -> list[str]:
    lines = ["## ETTh2 Strict Target-Clean Support", ""]
    lookback_filter = {int(item) for item in lookbacks} if lookbacks else None
    view_paths = []
    for path in sorted(views_dir.glob("ETTh2_L*_H*.csv")):
        match = VIEW_FILE_RE.match(path.name)
        if match is None:
            continue
        if lookback_filter is not None and int(match.group("lookback")) not in lookback_filter:
            continue
        view_paths.append(path)
    if not view_paths:
        lines.append(f"- 缺少 `{views_dir / 'ETTh2_L*_H*.csv'}`")
        lines.append("")
        return lines

    for path in view_paths:
        df = pd.read_csv(path, low_memory=False)
        test_rows = df[df["split_name"] == "test"].copy()
        strict_target_clean = int((test_rows["n_events_target"].fillna(0) == 0).sum())
        input_flagged_and_strict = int(
            ((test_rows["n_events_input"].fillna(0) > 0) & (test_rows["n_events_target"].fillna(0) == 0)).sum()
        )
        lookback = int(df["lookback"].iloc[0]) if "lookback" in df.columns and not df.empty else -1
        horizon = int(df["horizon"].iloc[0]) if "horizon" in df.columns and not df.empty else -1
        lines.append(
            f"- L={lookback} / H={horizon}: test windows={len(test_rows)}, strict target-clean={strict_target_clean}, "
            f"input-flagged & strict target-clean={input_flagged_and_strict}"
        )
    lines.extend(
        [
            "",
            "结论：如果任一主干对应 lookback 下 overall full-multivariate strict target-clean 仍为 0，011 必须切到 input-only paired + variable-stratified fallback。",
            "",
        ]
    )
    return lines


def solar_balanced_lines(manifest_path: Path) -> list[str]:
    lines = ["## solar_AL Balanced View Status", ""]
    if not manifest_path.exists():
        lines.append(f"- 缺少 `{manifest_path}`")
        lines.append("")
        return lines
    manifest = pd.read_csv(manifest_path, low_memory=False)
    sub = manifest[manifest["dataset_name"] == "solar_AL"].copy()
    if sub.empty:
        lines.append("- manifest 中没有 solar_AL 记录")
        lines.append("")
        return lines
    counts = sub.groupby(["split_name", "view_name"], dropna=False).size().reset_index(name="n")
    for split_name in ["train", "val", "test"]:
        split_rows = counts[counts["split_name"] == split_name]
        if split_rows.empty:
            continue
        summary = {
            str(row["view_name"]): int(row["n"])
            for _, row in split_rows.iterrows()
        }
        lines.append(f"- {split_name}: {summary}")
    lines.extend(
        [
            "",
            "判读规则：如果 `balanced` 与 `raw` 数量完全相同，它就仍然只是标签视图，不是真正改变分布的 balanced eval。",
            "",
        ]
    )
    return lines


def result_artifact_lines(results_dir: Path) -> list[str]:
    lines = ["## Existing Result Artifacts", ""]
    for rel_path in [
        "counterfactual_2x2.csv",
        "artifact_reliance_gap.csv",
        "worst_group_risk.csv",
        "ranking_instability.csv",
        "aef_results.csv",
        "aef_plus_results.csv",
        "aif_plus_results.csv",
    ]:
        path = results_dir / rel_path
        if not path.exists():
            lines.append(f"- `{rel_path}`: missing")
            continue
        df = pd.read_csv(path, low_memory=False)
        lines.append(f"- `{rel_path}`: {len(df)} rows")
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    views_dir = ROOT_DIR / Path(args.views_dir)
    manifest_path = ROOT_DIR / Path(args.manifest)
    results_dir = ROOT_DIR / Path(args.results_dir)
    out_path = ROOT_DIR / Path(args.out)

    lines = [
        "# 011 Step 0 Baseline Diagnosis",
        "",
        "这份报告固定 011 执行前的两个关键判断：`ETTh2` 是否发生 strict support collapse，以及 `solar_AL balanced` 是否仍然等于 raw。",
        "",
    ]
    lines.extend(etth2_strict_support_lines(views_dir, lookbacks=args.lookbacks))
    lines.extend(solar_balanced_lines(manifest_path))
    lines.extend(result_artifact_lines(results_dir))
    write_markdown(out_path, "\n".join(lines))


if __name__ == "__main__":
    main()
