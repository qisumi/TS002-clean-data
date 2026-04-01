from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from data import ROOT_DIR
from utils.experiment_profiles import FULL_EXPERIMENT_DATASETS, FULL_EXPERIMENT_HORIZONS, collect_required_lookbacks
from utils.module_runner import run_python_module


DEFAULT_ADDON_DATASETS = ["ETTm1", "ETTm2", "weather", "exchange_rate", "electricity"]
MISSING_CHECK_FILES = {
    "counterfactual": "counterfactual_2x2.csv",
    "aef": "aef_results.csv",
    "aef_plus": "aef_plus_results.csv",
    "aif_plus": "aif_plus_results.csv",
}
MERGE_FILES = [
    "counterfactual_2x2.csv",
    "artifact_reliance_gap.csv",
    "worst_group_risk.csv",
    "ranking_instability.csv",
    "aef_results.csv",
    "aef_control_results.csv",
    "aef_plus_results.csv",
    "aef_plus_artifact_reliance_gap.csv",
    "aef_plus_worst_group_risk.csv",
    "aif_plus_results.csv",
    "aif_plus_artifact_reliance_gap.csv",
    "aif_plus_worst_group_risk.csv",
    "aif_plus_ranking_instability.csv",
]


def log_progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [ADDON] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and merge missing add-on dataset experiments into the main results directory.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_ADDON_DATASETS)
    parser.add_argument("--counterfactual-config", default=str(Path("configs") / "counterfactual_eval.yaml"))
    parser.add_argument("--aef-config", default=str(Path("configs") / "aef.yaml"))
    parser.add_argument("--aef-plus-config", default=str(Path("configs") / "aef_plus.yaml"))
    parser.add_argument("--aif-plus-config", default=str(Path("configs") / "aif_plus.yaml"))
    parser.add_argument("--results-dir", default=str(Path("results")))
    parser.add_argument("--reports-dir", default=str(Path("reports")))
    parser.add_argument("--logs-dir", default=str(Path("logs")))
    parser.add_argument("--workspace-root", default=str(Path("results") / "addon_completion"))
    parser.add_argument("--force-datasets", action="store_true", help="Run the requested datasets even if result rows already exist.")
    parser.add_argument("--skip-handoff-refresh", action="store_true", help="Skip rebuilding handoff reports and ref_min after merge.")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def run_cli(module_name: str, args: list[str]) -> None:
    run_python_module(f"cli.{module_name}", args, cwd=ROOT_DIR, log=log_progress)


def dataset_rows(path: Path, dataset_name: str) -> int:
    if not path.exists():
        return 0
    df = pd.read_csv(path, low_memory=False)
    if "dataset_name" not in df.columns:
        return 0
    return int(df["dataset_name"].astype(str).eq(dataset_name).sum())


def resolve_missing_datasets(results_dir: Path, requested: list[str], force_datasets: bool) -> tuple[list[str], dict[str, dict[str, int]]]:
    detail: dict[str, dict[str, int]] = {}
    if force_datasets:
        for dataset_name in requested:
            detail[dataset_name] = {
                check_name: dataset_rows(results_dir / file_name, dataset_name)
                for check_name, file_name in MISSING_CHECK_FILES.items()
            }
        return requested, detail

    selected: list[str] = []
    for dataset_name in requested:
        counts = {
            check_name: dataset_rows(results_dir / file_name, dataset_name)
            for check_name, file_name in MISSING_CHECK_FILES.items()
        }
        detail[dataset_name] = counts
        if any(count <= 0 for count in counts.values()):
            selected.append(dataset_name)
    return selected, detail


def require_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required prerequisite file not found: {path}")


def resolve_required_lookbacks(counterfactual_config_path: Path) -> list[int]:
    payload = read_yaml(counterfactual_config_path)
    defaults = payload.get("defaults", {})
    runtime_defaults = dict(defaults.get("runtime", {}))
    if defaults.get("lookback") is not None:
        runtime_defaults["lookback_override"] = int(defaults["lookback"])
    lookbacks = set(
        collect_required_lookbacks(
            backbone_cfgs=[dict(item) for item in defaults.get("backbones", [])],
            datasets=FULL_EXPERIMENT_DATASETS,
            horizons=FULL_EXPERIMENT_HORIZONS,
            runtime_defaults=runtime_defaults,
        )
    )
    if "lookback" in defaults:
        lookbacks.add(int(defaults["lookback"]))
    return sorted(lookbacks)


def write_subset_config(base_config_path: Path, datasets: list[str], out_path: Path) -> Path:
    payload = read_yaml(base_config_path)
    payload.setdefault("defaults", {})
    payload["defaults"]["datasets"] = datasets
    write_yaml(out_path, payload)
    return out_path


def merge_dataset_csv(main_path: Path, addon_path: Path, datasets: list[str]) -> None:
    if not addon_path.exists():
        return
    addon_df = pd.read_csv(addon_path, low_memory=False)
    if addon_df.empty:
        return
    if main_path.exists():
        main_df = pd.read_csv(main_path, low_memory=False)
    else:
        main_df = pd.DataFrame(columns=addon_df.columns)

    if "dataset_name" in addon_df.columns:
        main_df = main_df[~main_df.get("dataset_name", pd.Series(dtype=str)).astype(str).isin(datasets)].copy()
    merged = pd.concat([main_df, addon_df], ignore_index=True, sort=False)
    sort_cols = [
        col
        for col in [
            "dataset_name",
            "lookback",
            "horizon",
            "backbone",
            "seed",
            "train_view_name",
            "train_view_token",
            "eval_view_name",
            "eval_view_token",
            "eval_protocol",
            "subset_name",
            "model_name",
        ]
        if col in merged.columns
    ]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)
    main_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(main_path, index=False)


def main() -> None:
    args = parse_args()
    requested = [str(item) for item in args.datasets]
    results_dir = ROOT_DIR / Path(args.results_dir)
    reports_dir = ROOT_DIR / Path(args.reports_dir)
    logs_dir = ROOT_DIR / Path(args.logs_dir)
    workspace_root = ROOT_DIR / Path(args.workspace_root)

    missing_datasets, detail = resolve_missing_datasets(results_dir, requested, force_datasets=bool(args.force_datasets))
    for dataset_name, counts in detail.items():
        log_progress(f"{dataset_name} coverage={counts}")
    if not missing_datasets:
        log_progress("all requested add-on datasets already have final result rows; nothing to run")
        return
    log_progress(f"selected datasets={missing_datasets}")

    workspace_root.mkdir(parents=True, exist_ok=True)
    temp_config_dir = logs_dir / "addon_completion" / "temp_configs"
    temp_reports_dir = reports_dir / "addon_completion"
    temp_logs_dir = logs_dir / "addon_completion"
    temp_reports_dir.mkdir(parents=True, exist_ok=True)
    temp_logs_dir.mkdir(parents=True, exist_ok=True)

    counterfactual_config = write_subset_config(
        ROOT_DIR / Path(args.counterfactual_config),
        datasets=missing_datasets,
        out_path=temp_config_dir / "counterfactual_eval.addon.yaml",
    )
    aef_config = write_subset_config(
        ROOT_DIR / Path(args.aef_config),
        datasets=missing_datasets,
        out_path=temp_config_dir / "aef.addon.yaml",
    )
    aef_plus_config = write_subset_config(
        ROOT_DIR / Path(args.aef_plus_config),
        datasets=missing_datasets,
        out_path=temp_config_dir / "aef_plus.addon.yaml",
    )
    aif_plus_config = write_subset_config(
        ROOT_DIR / Path(args.aif_plus_config),
        datasets=missing_datasets,
        out_path=temp_config_dir / "aif_plus.addon.yaml",
    )
    lookbacks = resolve_required_lookbacks(ROOT_DIR / Path(args.counterfactual_config))
    full_dataset_csv = ",".join(FULL_EXPERIMENT_DATASETS)

    # Rebuild shared upstream artifacts on the standard directories so the missing
    # datasets have end-to-end support from detection through window view construction.
    run_cli("build_dataset_registry", ["--datasets", full_dataset_csv])
    run_cli("generate_dataset_statistics", ["--datasets", full_dataset_csv])
    run_cli("detect_artifacts", ["--datasets", full_dataset_csv])
    run_cli("review_artifacts", ["--datasets", full_dataset_csv])
    run_cli(
        "merge_candidates_to_events",
        [
            "--stats-dir",
            "statistic_results",
            "--datasets",
            *FULL_EXPERIMENT_DATASETS,
            "--out-csv",
            "statistic_results/final_artifact_events.csv",
            "--out-md",
            "statistic_results/final_artifact_events.md",
        ]
    )
    run_cli(
        "render_event_merge_qa",
        [
            "--events",
            "statistic_results/final_artifact_events.csv",
            "--out-md",
            "reports/event_merge_visual_qa.md",
            "--n-per-dataset",
            "3",
        ]
    )
    for lookback in lookbacks:
        run_cli(
            "build_window_scores",
            [
                "--registry",
                "statistic_results/dataset_registry.csv",
                "--events",
                "statistic_results/final_artifact_events.csv",
                "--datasets",
                *FULL_EXPERIMENT_DATASETS,
                "--lookback",
                str(lookback),
                "--horizons",
                *[str(item) for item in FULL_EXPERIMENT_HORIZONS],
                "--out-dir",
                "statistic_results/window_scores",
                "--spec",
                "configs/view_specs.yaml",
            ]
        )
    run_cli(
        "build_eval_views",
        [
            "--spec",
            "configs/view_specs.yaml",
            "--scores-dir",
            "statistic_results/window_scores",
            "--out-dir",
            "statistic_results/window_views",
            "--events",
            "statistic_results/final_artifact_events.csv",
            "--manifest-out",
            "statistic_results/eval_view_manifest.csv",
            "--report-out",
            "reports/eval_view_design.md",
            "--datasets",
            *FULL_EXPERIMENT_DATASETS,
            "--lookbacks",
            *[str(item) for item in lookbacks],
            "--horizons",
            *[str(item) for item in FULL_EXPERIMENT_HORIZONS],
        ]
    )
    run_cli(
        "build_clean_view_qc",
        [
            "--events",
            "statistic_results/final_artifact_events.csv",
            "--views-dir",
            "statistic_results/window_views",
            "--out-md",
            "reports/clean_view_qc_report.md",
            "--out-csv",
            "reports/clean_view_support_summary.csv",
            "--datasets",
            *FULL_EXPERIMENT_DATASETS,
            "--lookbacks",
            *[str(item) for item in lookbacks],
            "--horizons",
            *[str(item) for item in FULL_EXPERIMENT_HORIZONS],
        ]
    )
    run_cli(
        "build_011_baseline_diagnosis",
        [
            "--views-dir",
            "statistic_results/window_views",
            "--manifest",
            "statistic_results/eval_view_manifest.csv",
            "--results-dir",
            "results",
            "--out",
            "reports/011_step0_baseline_diagnosis.md",
            "--lookbacks",
            *[str(item) for item in lookbacks],
        ]
    )

    for required_rel in [
        "statistic_results/dataset_registry.csv",
        "statistic_results/final_artifact_events.csv",
        "statistic_results/eval_view_manifest.csv",
        "reports/clean_view_support_summary.csv",
    ]:
        require_path(ROOT_DIR / required_rel)

    run_cli(
        "run_counterfactual_eval",
        [
            "--config",
            str(counterfactual_config.relative_to(ROOT_DIR)),
            "--manifest",
            "statistic_results/eval_view_manifest.csv",
            "--views-dir",
            "statistic_results/window_views",
            "--events",
            "statistic_results/final_artifact_events.csv",
            "--registry",
            "statistic_results/dataset_registry.csv",
            "--results-dir",
            str(workspace_root.relative_to(ROOT_DIR)),
            "--report-out",
            str((temp_reports_dir / "counterfactual_eval_summary.md").relative_to(ROOT_DIR)),
            "--setting-logs-dir",
            str((temp_logs_dir / "counterfactual_eval_settings").relative_to(ROOT_DIR)),
        ]
    )
    run_cli(
        "run_aef_baselines",
        [
            "--config",
            str(aef_config.relative_to(ROOT_DIR)),
            "--views-dir",
            "statistic_results/window_views",
            "--registry",
            "statistic_results/dataset_registry.csv",
            "--results-dir",
            str(workspace_root.relative_to(ROOT_DIR)),
            "--report-out",
            str((temp_reports_dir / "aef_summary.md").relative_to(ROOT_DIR)),
            "--baseline-results",
            str((workspace_root / "counterfactual_2x2.csv").relative_to(ROOT_DIR)),
        ]
    )
    run_cli(
        "run_aef_plus",
        [
            "--config",
            str(aef_plus_config.relative_to(ROOT_DIR)),
            "--views-dir",
            "statistic_results/window_views",
            "--registry",
            "statistic_results/dataset_registry.csv",
            "--events",
            "statistic_results/final_artifact_events.csv",
            "--baseline-results",
            str((workspace_root / "counterfactual_2x2.csv").relative_to(ROOT_DIR)),
            "--results-out",
            str((workspace_root / "aef_plus_results.csv").relative_to(ROOT_DIR)),
            "--window-errors-out",
            str((workspace_root / "aef_plus_window_errors.csv").relative_to(ROOT_DIR)),
            "--arg-out",
            str((workspace_root / "aef_plus_artifact_reliance_gap.csv").relative_to(ROOT_DIR)),
            "--wgr-out",
            str((workspace_root / "aef_plus_worst_group_risk.csv").relative_to(ROOT_DIR)),
            "--report-out",
            str((temp_reports_dir / "aef_plus_summary.md").relative_to(ROOT_DIR)),
        ]
    )
    run_cli(
        "run_aif_plus",
        [
            "--config",
            str(aif_plus_config.relative_to(ROOT_DIR)),
            "--views-dir",
            "statistic_results/window_views",
            "--registry",
            "statistic_results/dataset_registry.csv",
            "--events",
            "statistic_results/final_artifact_events.csv",
            "--support-summary",
            "reports/clean_view_support_summary.csv",
            "--baseline-results",
            str((workspace_root / "counterfactual_2x2.csv").relative_to(ROOT_DIR)),
            "--results-out",
            str((workspace_root / "aif_plus_results.csv").relative_to(ROOT_DIR)),
            "--window-errors-out",
            str((workspace_root / "aif_plus_window_errors.csv").relative_to(ROOT_DIR)),
            "--arg-out",
            str((workspace_root / "aif_plus_artifact_reliance_gap.csv").relative_to(ROOT_DIR)),
            "--wgr-out",
            str((workspace_root / "aif_plus_worst_group_risk.csv").relative_to(ROOT_DIR)),
            "--ri-out",
            str((workspace_root / "aif_plus_ranking_instability.csv").relative_to(ROOT_DIR)),
            "--report-out",
            str((temp_reports_dir / "aif_plus_summary.md").relative_to(ROOT_DIR)),
        ]
    )

    for file_name in MERGE_FILES:
        merge_dataset_csv(results_dir / file_name, workspace_root / file_name, missing_datasets)
        log_progress(f"merged {file_name}")

    run_cli(
        "build_011_baseline_diagnosis",
        [
            "--views-dir",
            "statistic_results/window_views",
            "--manifest",
            "statistic_results/eval_view_manifest.csv",
            "--results-dir",
            "results",
            "--out",
            "reports/011_step0_baseline_diagnosis.md",
            "--lookbacks",
            *[str(item) for item in lookbacks],
        ]
    )

    if not args.skip_handoff_refresh:
        run_cli(
            "build_unified_leaderboard_appendix",
            [
                "--results-dir",
                str(results_dir.relative_to(ROOT_DIR)),
                "--reports-dir",
                str(reports_dir.relative_to(ROOT_DIR)),
                "--support-summary",
                str((reports_dir / "clean_view_support_summary.csv").relative_to(ROOT_DIR)),
            ]
        )
        run_cli(
            "build_handoff_reports",
            [
                "--results-dir",
                str(results_dir.relative_to(ROOT_DIR)),
                "--reports-dir",
                str(reports_dir.relative_to(ROOT_DIR)),
                "--stats-dir",
                "statistic_results",
            ]
        )
        run_cli("organize_handoff_outputs", ["--manifest-out", "cleanup_manifest.md"])

    log_progress("addon experiment completion finished")


if __name__ == "__main__":
    main()
