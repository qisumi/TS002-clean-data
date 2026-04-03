from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data import ROOT_DIR, write_markdown


RESULT_EXCLUDE = {
    "aef_control_window_errors.csv",
    "aef_plus_window_errors.csv",
}

STAT_EXCLUDE = {
    "eval_view_manifest.csv",
    "final_artifact_metadata.csv",
}

OMITTED_ITEMS = [
    ("results/aef_control_window_errors.csv", "窗口级误差明细，69.6 MB，过大且不适合网页版直接分析。"),
    ("results/aef_plus_window_errors.csv", "窗口级误差明细，121.8 MB，过大且不适合网页版直接分析。"),
    ("statistic_results/eval_view_manifest.csv", "view-level 主清单，904.6 MB，体积过大。"),
    ("statistic_results/final_artifact_metadata.csv", "与 `final_artifact_events.csv` 当前内容完全重复，避免重复上传。"),
    ("statistic_results/window_scores/", "逐窗口分数明细目录，体积大，保留在仓库原位置。"),
    ("statistic_results/window_views/", "逐窗口 view 明细目录，体积大，保留在仓库原位置。"),
    ("logs/newplan/children/", "子任务分片日志，噪声较多，不影响结果分析。"),
    ("logs/newplan/08-run-aif-plus.*", "旧命名遗留日志，不属于本次标准 001-009 完成链路。"),
]

AIF_PLUS_PQT_ARTIFACT_REL_PATHS = [
    "results/aif_plus_pqt_results.csv",
    "results/aif_plus_pqt_window_errors.csv",
    "results/aif_plus_pqt_artifact_reliance_gap.csv",
    "results/aif_plus_pqt_worst_group_risk.csv",
    "results/aif_plus_pqt_ranking_instability.csv",
    "reports/aif_plus_pqt_summary.md",
]

PROMPT_TEXT = """请先阅读 `README_BUNDLE.md`，再结合以下文件做结果分析：

1. `reports/011_next_step_results_report.md`
2. `reports/011_step0_baseline_diagnosis.md`
3. `reports/011_ETTh1_results_page.md`
4. `reports/011_weather_results_page.md`
5. `reports/011_support_bundle_manifest.md`
6. `results/unified_leaderboard_summary.csv`
7. `results/counterfactual_2x2.csv`
8. `results/artifact_reliance_gap.csv`
9. `results/worst_group_risk.csv`
10. `results/ranking_instability.csv`
11. `statistic_results/final_artifact_events.csv`
12. `reports/clean_view_support_summary.csv`

请输出：

- 核心研究结论，按数据集和方法分组总结。
- 哪些结论最稳健，哪些结论可能受脏数据或视图选择影响。
- `ETTh2 / solar_AL / ETTh1 / weather` 的重点发现与风险。
- 如果我要把这些结果写成论文或汇报，最值得保留的表述和图表建议。
- 如果还需要补实验，请给出优先级最高的 3-5 个下一步。
"""


@dataclass(frozen=True)
class BundleFile:
    rel_path: str
    category: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a compact ChatGPT upload bundle from completed pipeline outputs.")
    parser.add_argument("--bundle-dir", default="chatgpt_upload_bundle")
    parser.add_argument("--archive-name", default="chatgpt_upload_bundle.zip")
    return parser.parse_args()


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def csv_row_count(path: Path) -> int:
    return len(pd.read_csv(path, low_memory=False))


def collect_report_files(root_dir: Path) -> list[BundleFile]:
    notes = {
        "011_next_step_results_report.md": "总入口，适合让 ChatGPT 先读。",
        "011_step0_baseline_diagnosis.md": "基线层面的诊断总结。",
        "011_ETTh1_results_page.md": "ETTh1 单页汇总。",
        "011_weather_results_page.md": "weather 单页汇总。",
        "011_support_bundle_manifest.md": "主文件存在性和 CSV 行数清单。",
        "clean_view_qc_report.md": "clean/intervened view 的 QC 报告。",
        "clean_view_support_summary.csv": "各数据集可用 clean/intervened window 支持度。",
        "artifact_event_summary.md": "artifact event 层面的汇总判断。",
        "event_merge_visual_qa.md": "event merge 的抽样 QA。",
        "eval_view_design.md": "view 设计说明。",
        "counterfactual_eval_summary.md": "counterfactual 结果摘要。",
        "aef_summary.md": "AEF 汇总。",
        "aef_plus_summary.md": "AEF-Plus 汇总。",
        "aif_plus_summary.md": "AIF-Plus 汇总。",
        "aif_plus_pqt_summary.md": "AIF-Plus-PQT 汇总。",
        "unified_leaderboard_appendix.md": "统一 leaderboard 附录。",
        "addon_dataset_status_summary.md": "附加数据集完成度状态汇总。",
        "etth2_variable_stratified_eval.md": "ETTh2 变量分层分析说明。",
    }
    out: list[BundleFile] = []
    for path in sorted((root_dir / "reports").glob("*")):
        if not path.is_file():
            continue
        out.append(BundleFile(str(path.relative_to(root_dir)), "report", notes.get(path.name, "报告或说明文件。")))
    return out


def collect_result_files(root_dir: Path) -> list[BundleFile]:
    notes = {
        "counterfactual_2x2.csv": "主实验聚合表，包含 raw/clean/intervened 设定结果。",
        "artifact_reliance_gap.csv": "ARG 指标汇总。",
        "worst_group_risk.csv": "WGR 指标汇总。",
        "ranking_instability.csv": "RI 指标汇总。",
        "etth2_channel_support.csv": "ETTh2 channel support 诊断。",
        "etth2_variable_stratified_eval.csv": "ETTh2 变量分层实验结果。",
        "aef_results.csv": "AEF-Weak 结果。",
        "aef_control_results.csv": "AEF-Control 结果。",
        "aef_plus_results.csv": "AEF-Plus 结果。",
        "aef_plus_artifact_reliance_gap.csv": "AEF-Plus 的 ARG 汇总。",
        "aef_plus_worst_group_risk.csv": "AEF-Plus 的 WGR 汇总。",
        "aif_plus_results.csv": "AIF-Plus 结果。",
        "aif_plus_artifact_reliance_gap.csv": "AIF-Plus 的 ARG 汇总。",
        "aif_plus_worst_group_risk.csv": "AIF-Plus 的 WGR 汇总。",
        "aif_plus_ranking_instability.csv": "AIF-Plus 的 RI 汇总。",
        "aif_plus_pqt_results.csv": "AIF-Plus-PQT 结果。",
        "aif_plus_pqt_window_errors.csv": "AIF-Plus-PQT 窗口级误差明细。",
        "aif_plus_pqt_artifact_reliance_gap.csv": "AIF-Plus-PQT 的 ARG 汇总。",
        "aif_plus_pqt_worst_group_risk.csv": "AIF-Plus-PQT 的 WGR 汇总。",
        "aif_plus_pqt_ranking_instability.csv": "AIF-Plus-PQT 的 RI 汇总。",
        "unified_leaderboard_rows.csv": "统一 leaderboard 明细行。",
        "unified_leaderboard_summary.csv": "统一 leaderboard 摘要。",
        "unified_significance.csv": "显著性对比汇总。",
        "unified_error_distribution.csv": "误差分布聚合表。",
    }
    out: list[BundleFile] = []
    for path in sorted((root_dir / "results").glob("*")):
        if not path.is_file() or path.name in RESULT_EXCLUDE:
            continue
        out.append(BundleFile(str(path.relative_to(root_dir)), "result", notes.get(path.name, "结果聚合 CSV。")))
    return out


def collect_stat_files(root_dir: Path) -> list[BundleFile]:
    notes = {
        "artifact_candidates.csv": "artifact 候选级别明细。",
        "artifact_taxonomy.md": "artifact taxonomy 说明。",
        "candidate_to_event_map.csv": "候选到事件的映射表。",
        "dataset_registry.csv": "数据集注册表。",
        "dataset_registry_snapshot.csv": "数据集注册表快照。",
        "dataset_statistics.csv": "基础统计信息。",
        "final_artifact_events.csv": "最终 artifact event 主表。",
        "final_artifact_events.md": "最终 artifact event 的文字版摘要。",
        "final_artifact_metadata.md": "最终 metadata 的文字版摘要。",
        "manual_review_notes.md": "人工 review 说明。",
        "solar_AL_phase_annotations.csv": "solar_AL 相位注释。",
        "solar_AL_phase_profile.csv": "solar_AL 相位 profile。",
    }
    out: list[BundleFile] = []
    for path in sorted((root_dir / "statistic_results").glob("*")):
        if not path.is_file() or path.name in STAT_EXCLUDE:
            continue
        note = notes.get(path.name)
        if note is None:
            if path.name.endswith("_flagged.csv"):
                note = "数据集级脏数据候选标记表。"
            elif path.name.endswith("_event_metadata.csv"):
                note = "数据集级合并事件 metadata。"
            else:
                note = "统计阶段输出文件。"
        out.append(BundleFile(str(path.relative_to(root_dir)), "stat", note))
    return out


def collect_log_files(root_dir: Path) -> list[BundleFile]:
    pattern = re.compile(r"^\d{3}-.*\.(log|ok)$")
    out: list[BundleFile] = []
    log_dir = root_dir / "logs" / "newplan"
    for path in sorted(log_dir.glob("*")):
        if not path.is_file() or not pattern.match(path.name):
            continue
        note = "标准链路日志。" if path.suffix == ".log" else "标准链路完成标记。"
        out.append(BundleFile(str(path.relative_to(root_dir)), "log", note))
    return out


def collect_bundle_files(root_dir: Path) -> list[BundleFile]:
    files = []
    files.extend(collect_report_files(root_dir))
    files.extend(collect_result_files(root_dir))
    files.extend(collect_stat_files(root_dir))
    files.extend(collect_log_files(root_dir))
    return files


def copy_files(root_dir: Path, bundle_dir: Path, bundle_files: list[BundleFile]) -> None:
    for item in bundle_files:
        src = root_dir / item.rel_path
        dst = bundle_dir / item.rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_run_status_section(root_dir: Path) -> list[str]:
    lines = [
        "## Run Status",
        "",
    ]
    log_dir = root_dir / "logs" / "newplan"
    steps = [
        "001-build-registry-and-stats",
        "002-detect-review-and-merge-events",
        "003-build-window-scores-views-and-qc",
        "004-run-counterfactual-mainboards",
        "005-run-etth2-support-boundary",
        "006-run-aef-baselines",
        "007-run-aef-plus",
        "008-run-aif-plus",
        "009-build-final-reports",
    ]
    for step in steps:
        ok = (log_dir / f"{step}.ok").exists()
        log_exists = (log_dir / f"{step}.log").exists()
        lines.append(f"- `{step}`: ok_marker={ok}, log_present={log_exists}")
    if (log_dir / "08-run-aif-plus.failed").exists():
        lines.append("- 发现 `logs/newplan/08-run-aif-plus.failed` 旧遗留文件，未纳入上传包；标准 `008-run-aif-plus.ok` 已存在。")
    lines.append("")
    return lines


def build_inventory_section(root_dir: Path, bundle_files: list[BundleFile]) -> list[str]:
    lines = [
        "## Included Files",
        "",
        "| path | category | size | rows | note |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in bundle_files:
        path = root_dir / item.rel_path
        rows = ""
        if path.suffix.lower() == ".csv":
            rows = str(csv_row_count(path))
        lines.append(
            f"| `{item.rel_path}` | {item.category} | {format_size(path.stat().st_size)} | {rows or '-'} | {item.note} |"
        )
    lines.append("")
    return lines


def build_omitted_section() -> list[str]:
    lines = [
        "## Omitted Large Or Redundant Items",
        "",
    ]
    for rel_path, reason in OMITTED_ITEMS:
        lines.append(f"- `{rel_path}`: {reason}")
    lines.append("")
    return lines


def build_readme(root_dir: Path, bundle_files: list[BundleFile]) -> str:
    lines = [
        "# ChatGPT Upload Bundle",
        "",
        "这个目录是从已完成的 `plan/010-run-all.sh` 产物中筛出的精简上传包，目标是让 ChatGPT 网页版能直接读取并分析核心结果，",
        "同时保留足够的追溯文件而不把 2GB+ 的窗口级大表一起塞进去。",
        "",
        "## Suggested Reading Order",
        "",
        "- 先读 `reports/011_next_step_results_report.md`。",
        "- 再读 `reports/011_step0_baseline_diagnosis.md` 与 `reports/011_support_bundle_manifest.md`。",
        "- 然后结合 `results/unified_leaderboard_summary.csv`、`results/counterfactual_2x2.csv`、`results/artifact_reliance_gap.csv`、`results/worst_group_risk.csv`、`results/ranking_instability.csv`。",
        "- 如果要追溯事件来源，再看 `statistic_results/final_artifact_events.csv`、各数据集 `*_event_metadata.csv`、`*_flagged.csv`。",
        "",
    ]
    lines.extend(build_run_status_section(root_dir))
    lines.extend(build_inventory_section(root_dir, bundle_files))
    lines.extend(build_omitted_section())
    lines.extend(
        [
            "## Suggested Prompt",
            "",
            "更完整的提示词见 `CHATGPT_PROMPT.md`。",
            "",
        ]
    )
    return "\n".join(lines)


def write_zip(bundle_dir: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(bundle_dir.rglob("*")):
            if not path.is_file():
                continue
            zf.write(path, arcname=str(path.relative_to(bundle_dir.parent)))


def existing_paths(root_dir: Path, rel_paths: list[str]) -> list[Path]:
    return [root_dir / rel_path for rel_path in rel_paths if (root_dir / rel_path).exists()]


def write_selected_zip(root_dir: Path, archive_path: Path, paths: list[Path]) -> None:
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(paths):
            zf.write(path, arcname=str(path.relative_to(root_dir)))


def main() -> None:
    args = parse_args()
    root_dir = ROOT_DIR
    bundle_dir = root_dir / args.bundle_dir
    archive_path = root_dir / args.archive_name

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    bundle_files = collect_bundle_files(root_dir)
    copy_files(root_dir, bundle_dir, bundle_files)

    write_markdown(bundle_dir / "README_BUNDLE.md", build_readme(root_dir, bundle_files))
    write_markdown(bundle_dir / "CHATGPT_PROMPT.md", PROMPT_TEXT)

    write_zip(bundle_dir, archive_path)

    aif_plus_pqt_paths = existing_paths(root_dir, AIF_PLUS_PQT_ARTIFACT_REL_PATHS)
    if len(aif_plus_pqt_paths) == len(AIF_PLUS_PQT_ARTIFACT_REL_PATHS):
        write_selected_zip(root_dir, root_dir / "011-aif-plus-pqt-artifacts.zip", aif_plus_pqt_paths)


if __name__ == "__main__":
    main()
