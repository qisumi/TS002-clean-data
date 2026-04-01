from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from data import ROOT_DIR, write_markdown


REF_MIN_FILES = [
    "reports/011_next_step_results_report.md",
    "reports/011_step0_baseline_diagnosis.md",
    "reports/clean_view_qc_report.md",
    "reports/clean_view_support_summary.csv",
    "reports/011_ETTh1_results_page.md",
    "reports/011_weather_results_page.md",
    "reports/addon_dataset_status_summary.md",
    "plan/011-next_step_plan_report_v2_with_algorithms.md",
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
    "statistic_results/artifact_taxonomy.md",
    "statistic_results/dataset_registry.csv",
    "statistic_results/final_artifact_events.csv",
    "statistic_results/final_artifact_events.md",
    "statistic_results/final_artifact_metadata.csv",
    "statistic_results/final_artifact_metadata.md",
    "statistic_results/solar_AL_phase_annotations.csv",
    "statistic_results/solar_AL_phase_profile.csv",
]

ARCHIVE_SCAN_ROOTS = [
    "plan",
    "reports",
    "results",
    "statistic_results",
    "figures/feature_plots",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy standard outputs into ref_min/archive_not_for_handoff/delete_now.")
    parser.add_argument("--clean-existing", action="store_true", help="Remove existing handoff directories before repopulating.")
    parser.add_argument("--manifest-out", default=str(ROOT_DIR / "cleanup_manifest.md"))
    return parser.parse_args()


def collect_all_files(root_dir: Path, rel_root: str) -> set[str]:
    source_dir = root_dir / rel_root
    if not source_dir.exists():
        return set()
    return {
        str(path.relative_to(root_dir))
        for path in source_dir.rglob("*")
        if path.is_file()
    }


def collect_delete_files(root_dir: Path) -> set[str]:
    delete_paths: set[str] = set()
    logs_dir = root_dir / "logs"
    if logs_dir.exists():
        delete_paths.update(collect_all_files(root_dir, "logs"))

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        rel_path_obj = path.relative_to(root_dir)
        if rel_path_obj.parts and rel_path_obj.parts[0] in {"ref_min", "archive_not_for_handoff", "delete_now"}:
            continue
        rel_path = str(rel_path_obj)
        if "__pycache__" in path.parts:
            delete_paths.add(rel_path)
        elif rel_path.startswith("reports/_smoke_") and path.suffix.lower() == ".md":
            delete_paths.add(rel_path)
        elif rel_path.startswith("results/_smoke"):
            delete_paths.add(rel_path)
        elif rel_path == "temp_weather_011.sh":
            delete_paths.add(rel_path)
    return delete_paths


def copy_relative_files(root_dir: Path, rel_paths: set[str], target_root: Path) -> int:
    copied = 0
    for rel_path in sorted(rel_paths):
        source = root_dir / rel_path
        if not source.exists() or not source.is_file():
            continue
        destination = target_root / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    return copied


def build_manifest(
    ref_min_files: set[str],
    archive_files: set[str],
    delete_files: set[str],
    missing_ref_files: list[str],
) -> str:
    lines = [
        "# Cleanup Manifest",
        "",
        "本清单记录标准目录产物二次整理到 handoff 目录后的映射结果。",
        "",
        f"- ref_min files: {len(ref_min_files)}",
        f"- archive_not_for_handoff files: {len(archive_files)}",
        f"- delete_now files: {len(delete_files)}",
        "",
        "## Missing ref_min Files",
        "",
    ]
    if missing_ref_files:
        for rel_path in missing_ref_files:
            lines.append(f"- `{rel_path}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- 整理逻辑为复制，不直接把实验脚本的输出目录改写成 handoff 目录。",
            "- `archive_not_for_handoff` 收纳 `plan/reports/results/statistic_results` 中除 ref_min 与 delete_now 以外的其余文件。",
            "- `delete_now` 收纳 `logs/`、`__pycache__/`、`_smoke*` 和 `temp_weather_011.sh`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root_dir = ROOT_DIR
    ref_min_dir = root_dir / "ref_min"
    archive_dir = root_dir / "archive_not_for_handoff"
    delete_dir = root_dir / "delete_now"

    if args.clean_existing:
        for target_dir in [ref_min_dir, archive_dir, delete_dir]:
            if target_dir.exists():
                shutil.rmtree(target_dir)

    ref_min_set = {rel_path for rel_path in REF_MIN_FILES if (root_dir / rel_path).exists()}
    missing_ref = sorted(set(REF_MIN_FILES) - ref_min_set)
    delete_set = collect_delete_files(root_dir)

    archive_candidates: set[str] = set()
    for rel_root in ARCHIVE_SCAN_ROOTS:
        archive_candidates.update(collect_all_files(root_dir, rel_root))
    archive_set = {
        rel_path
        for rel_path in archive_candidates
        if rel_path not in ref_min_set and rel_path not in delete_set
    }

    copy_relative_files(root_dir, ref_min_set, ref_min_dir)
    copy_relative_files(root_dir, archive_set, archive_dir)
    copy_relative_files(root_dir, delete_set, delete_dir)

    manifest_out = Path(args.manifest_out)
    if not manifest_out.is_absolute():
        manifest_out = root_dir / manifest_out
    write_markdown(
        manifest_out,
        build_manifest(
            ref_min_files=ref_min_set,
            archive_files=archive_set,
            delete_files=delete_set,
            missing_ref_files=missing_ref,
        ),
    )


if __name__ == "__main__":
    main()
