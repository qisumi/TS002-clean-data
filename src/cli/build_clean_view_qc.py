from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

from data import ROOT_DIR, ensure_project_directories, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render clean/intervened view support QC.")
    parser.add_argument("--events", default=str(Path("statistic_results") / "final_artifact_events.csv"))
    parser.add_argument("--views-dir", default=str(Path("statistic_results") / "window_views"))
    parser.add_argument("--out-md", default=str(ROOT_DIR / "reports" / "clean_view_qc_report.md"))
    parser.add_argument("--out-csv", default=str(ROOT_DIR / "reports" / "clean_view_support_summary.csv"))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--lookbacks", nargs="+", type=int, default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    return parser.parse_args()


VIEW_FILE_RE = re.compile(r"^(?P<dataset>.+)_L(?P<lookback>\d+)_H(?P<horizon>\d+)\.csv$")


def view_path_matches(
    path: Path,
    datasets: set[str] | None,
    lookbacks: set[int] | None,
    horizons: set[int] | None,
) -> bool:
    match = VIEW_FILE_RE.match(path.name)
    if match is None:
        return False
    dataset_name = str(match.group("dataset"))
    lookback = int(match.group("lookback"))
    horizon = int(match.group("horizon"))
    if datasets is not None and dataset_name not in datasets:
        return False
    if lookbacks is not None and lookback not in lookbacks:
        return False
    if horizons is not None and horizon not in horizons:
        return False
    return True


def load_view_files(
    views_dir: Path,
    datasets: list[str] | None = None,
    lookbacks: list[int] | None = None,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    dataset_filter = {str(item) for item in datasets} if datasets else None
    lookback_filter = {int(item) for item in lookbacks} if lookbacks else None
    horizon_filter = {int(item) for item in horizons} if horizons else None
    for path in sorted(views_dir.glob("*.csv")):
        if not view_path_matches(path, datasets=dataset_filter, lookbacks=lookback_filter, horizons=horizon_filter):
            continue
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_support_summary(view_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        view_df.groupby(["dataset_name", "lookback", "horizon", "split_name", "view_status"], dropna=False)
        .agg(
            raw_windows=("is_raw_view", "sum"),
            anchor_clean_windows=("is_anchor_clean_view", "sum"),
            conservative_clean_windows=("is_conservative_clean_view", "sum"),
            intervened_windows=("is_intervened_view", "sum"),
            phase_balanced_windows=("is_phase_balanced_view", "sum"),
            active_only_windows=("is_active_only_view", "sum"),
            daytime_only_windows=("is_daytime_only_view", "sum"),
        )
        .reset_index()
    )
    summary["anchor_ratio"] = (summary["anchor_clean_windows"] / summary["raw_windows"].clip(lower=1)).round(6)
    summary["conservative_ratio"] = (summary["conservative_clean_windows"] / summary["raw_windows"].clip(lower=1)).round(6)
    summary["intervened_ratio"] = (summary["intervened_windows"] / summary["raw_windows"].clip(lower=1)).round(6)
    return summary


def build_qc_markdown(events: pd.DataFrame, view_df: pd.DataFrame, support_df: pd.DataFrame) -> str:
    lines = [
        "# Clean View QC Report",
        "",
        f"- 事件级输入数: {len(events)}",
        f"- 窗口级输入数: {len(view_df)}",
        "",
    ]

    for dataset_name, group in support_df.groupby("dataset_name", dropna=False):
        lines.extend([f"## {dataset_name}", ""])
        for row in group.sort_values(["lookback", "horizon", "split_name"]).itertuples(index=False):
            lines.extend(
                [
                    f"### L={int(row.lookback)} / H={int(row.horizon)} / split={row.split_name}",
                    "",
                    f"- view_status: `{row.view_status}`",
                    f"- raw / anchor / conservative / intervened: {int(row.raw_windows)} / {int(row.anchor_clean_windows)} / {int(row.conservative_clean_windows)} / {int(row.intervened_windows)}",
                    f"- anchor_ratio / conservative_ratio / intervened_ratio: {float(row.anchor_ratio):.3f} / {float(row.conservative_ratio):.3f} / {float(row.intervened_ratio):.3f}",
                ]
            )
            if dataset_name == "solar_AL":
                lines.append(
                    f"- balanced / active_only / daytime_only: {int(row.phase_balanced_windows)} / {int(row.active_only_windows)} / {int(row.daytime_only_windows)}"
                )
            lines.append("")

        fallback_rows = group[group["view_status"].isin(["anchor_only", "fallback_to_intervention", "intervention_primary"])]
        if not fallback_rows.empty:
            lines.append(f"- 自动降级: {fallback_rows[['horizon', 'split_name', 'view_status']].to_dict(orient='records')}")
            lines.append("")

    solar_rows = support_df[support_df["dataset_name"] == "solar_AL"]
    if not solar_rows.empty:
        lines.extend(
            [
                "## Solar Checks",
                "",
                "- night_zero_band 只通过 phase-aware 视图处理，不进入 blind repair。",
                f"- solar view_status 分布: {dict(solar_rows['view_status'].value_counts())}",
                "",
            ]
        )

    anomalies: list[str] = []
    if (view_df["is_intervened_view"] == 0).all():
        anomalies.append("所有窗口都被排除出 intervened_view，需要人工确认 recoverability 规则是否过严。")
    if (view_df["dataset_name"].eq("solar_AL") & view_df["is_active_only_view"].eq(0)).all():
        anomalies.append("solar_AL 没有 active_only 窗口，需要复查 active 相位阈值。")
    if not anomalies:
        anomalies.append("未发现需要立即人工复查的结构性异常。")

    lines.extend(["## 人工复查项", ""])
    lines.extend([f"- {item}" for item in anomalies])
    lines.append("")
    return "\n".join(lines)


def run_build_clean_view_qc(
    events_path: Path,
    views_dir: Path,
    out_md: Path,
    out_csv: Path,
    datasets: list[str] | None = None,
    lookbacks: list[int] | None = None,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    ensure_project_directories()
    events = pd.read_csv(events_path)
    view_df = load_view_files(
        views_dir,
        datasets=datasets,
        lookbacks=lookbacks,
        horizons=horizons,
    )
    support_df = build_support_summary(view_df) if not view_df.empty else pd.DataFrame()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    support_df.to_csv(out_csv, index=False)
    write_markdown(out_md, build_qc_markdown(events, view_df, support_df))
    return support_df


def main() -> None:
    args = parse_args()
    run_build_clean_view_qc(
        events_path=Path(args.events),
        views_dir=Path(args.views_dir),
        out_md=Path(args.out_md),
        out_csv=Path(args.out_csv),
        datasets=args.datasets,
        lookbacks=args.lookbacks,
        horizons=args.horizons,
    )


if __name__ == "__main__":
    main()
