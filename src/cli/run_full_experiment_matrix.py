from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from data import ROOT_DIR
from utils.experiment_profiles import FULL_EXPERIMENT_DATASETS, FULL_EXPERIMENT_HORIZONS, collect_required_lookbacks
from utils.module_runner import run_python_module, with_src_pythonpath


def log_progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [FULL] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full multi-dataset experiment matrix and then organize outputs.")
    parser.add_argument("--datasets", nargs="+", default=FULL_EXPERIMENT_DATASETS)
    parser.add_argument("--horizons", nargs="+", type=int, default=FULL_EXPERIMENT_HORIZONS)
    parser.add_argument("--counterfactual-config", default=str(Path("configs") / "counterfactual_eval.yaml"))
    parser.add_argument("--aef-config", default=str(Path("configs") / "aef.yaml"))
    parser.add_argument("--aef-plus-config", default=str(Path("configs") / "aef_plus.yaml"))
    parser.add_argument("--aif-plus-config", default=str(Path("configs") / "aif_plus.yaml"))
    parser.add_argument("--conda-env", default="zzq", help="Conda env used for all child pipeline stages.")
    parser.add_argument("--skip-conda-run", action="store_true", help="Use the current interpreter instead of conda run.")
    parser.add_argument("--skip-organize", action="store_true")
    parser.add_argument("--clean-handoff", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_python_launcher(conda_env: str, skip_conda_run: bool) -> list[str]:
    if skip_conda_run:
        return [sys.executable]

    current_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    current_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if current_env == conda_env and current_prefix and Path(sys.executable).is_file():
        executable_path = Path(sys.executable).resolve()
        prefix_path = Path(current_prefix).resolve()
        try:
            executable_path.relative_to(prefix_path)
            return [sys.executable]
        except ValueError:
            pass

    conda_bin = shutil.which("conda")
    if conda_bin is None:
        raise FileNotFoundError("conda not found in PATH; cannot ensure the requested conda environment")

    proc = subprocess.run(
        [conda_bin, "env", "list", "--json"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout or "{}")
    env_names = {Path(env_path).name for env_path in payload.get("envs", [])}
    if conda_env not in env_names:
        raise ValueError(f"conda env `{conda_env}` not found; available envs={sorted(env_names)}")
    return [conda_bin, "run", "--no-capture-output", "-n", conda_env, "python"]


def log_python_launcher(launcher: list[str]) -> None:
    probe = subprocess.run(
        [
            *launcher,
            "-c",
            "import os, sys; "
            "print(f'python={sys.executable}'); "
            "print(f'conda_env={os.getenv(\"CONDA_DEFAULT_ENV\", \"\") or \"<unset>\"}'); "
            "print(f'prefix={sys.prefix}')",
        ],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
        env=with_src_pythonpath(root_dir=ROOT_DIR),
    )
    for line in (probe.stdout or "").splitlines():
        if line.strip():
            log_progress(line.strip())


def run_cli(module_name: str, args: list[str], launcher: list[str]) -> None:
    run_python_module(f"cli.{module_name}", args, launcher=launcher, cwd=ROOT_DIR, log=log_progress)


def resolve_required_lookbacks(counterfactual_config_path: Path, datasets: list[str], horizons: list[int]) -> list[int]:
    config = load_yaml(counterfactual_config_path)
    defaults = config.get("defaults", {})
    runtime_defaults = dict(defaults.get("runtime", {}))
    if defaults.get("lookback") is not None:
        runtime_defaults["lookback_override"] = int(defaults["lookback"])
    lookbacks = set(
        collect_required_lookbacks(
            backbone_cfgs=[dict(item) for item in defaults.get("backbones", [])],
            datasets=datasets,
            horizons=horizons,
            runtime_defaults=runtime_defaults,
        )
    )
    if "lookback" in defaults:
        lookbacks.add(int(defaults["lookback"]))
    return sorted(lookbacks)


def main() -> None:
    args = parse_args()
    datasets = [str(item) for item in args.datasets]
    horizons = [int(item) for item in args.horizons]
    dataset_csv = ",".join(datasets)
    counterfactual_config = ROOT_DIR / Path(args.counterfactual_config)
    aef_config = ROOT_DIR / Path(args.aef_config)
    aef_plus_config = ROOT_DIR / Path(args.aef_plus_config)
    aif_plus_config = ROOT_DIR / Path(args.aif_plus_config)
    lookbacks = resolve_required_lookbacks(counterfactual_config, datasets=datasets, horizons=horizons)
    launcher = resolve_python_launcher(conda_env=str(args.conda_env), skip_conda_run=bool(args.skip_conda_run))

    log_progress(f"datasets={datasets}")
    log_progress(f"horizons={horizons}")
    log_progress(f"required_lookbacks={lookbacks}")
    log_python_launcher(launcher)

    run_cli("build_dataset_registry", ["--datasets", dataset_csv], launcher=launcher)
    run_cli("generate_dataset_statistics", ["--datasets", dataset_csv], launcher=launcher)
    run_cli("detect_artifacts", ["--datasets", dataset_csv], launcher=launcher)
    run_cli("review_artifacts", ["--datasets", dataset_csv], launcher=launcher)
    run_cli(
        "merge_candidates_to_events",
        [
            "--stats-dir",
            "statistic_results",
            "--datasets",
            *datasets,
            "--out-csv",
            "statistic_results/final_artifact_events.csv",
            "--out-md",
            "statistic_results/final_artifact_events.md",
        ],
        launcher=launcher,
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
        ],
        launcher=launcher,
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
                *datasets,
                "--lookback",
                str(lookback),
                "--horizons",
                *[str(item) for item in horizons],
                "--out-dir",
                "statistic_results/window_scores",
                "--spec",
                "configs/view_specs.yaml",
            ],
            launcher=launcher,
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
            *datasets,
            "--lookbacks",
            *[str(item) for item in lookbacks],
            "--horizons",
            *[str(item) for item in horizons],
        ],
        launcher=launcher,
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
            *datasets,
            "--lookbacks",
            *[str(item) for item in lookbacks],
            "--horizons",
            *[str(item) for item in horizons],
        ],
        launcher=launcher,
    )
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
            "results",
            "--report-out",
            "reports/counterfactual_eval_summary.md",
            "--setting-logs-dir",
            "logs/counterfactual_eval_settings",
        ],
        launcher=launcher,
    )

    if "ETTh2" in datasets:
        run_cli(
            "build_etth2_channel_support",
            [
                "--events",
                "statistic_results/final_artifact_events.csv",
                "--views-dir",
                "statistic_results/window_views",
                "--out",
                "results/etth2_channel_support.csv",
                "--lookbacks",
                ",".join(str(item) for item in lookbacks),
                "--horizons",
                ",".join(str(item) for item in horizons),
            ],
            launcher=launcher,
        )
        run_cli(
            "run_etth2_variable_stratified_eval",
            [
                "--config",
                str(counterfactual_config.relative_to(ROOT_DIR)),
                "--registry",
                "statistic_results/dataset_registry.csv",
                "--events",
                "statistic_results/final_artifact_events.csv",
                "--views-dir",
                "statistic_results/window_views",
                "--support-csv",
                "results/etth2_channel_support.csv",
                "--results-out",
                "results/etth2_variable_stratified_eval.csv",
                "--report-out",
                "reports/etth2_variable_stratified_eval.md",
            ],
            launcher=launcher,
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
            "results",
            "--report-out",
            "reports/aef_summary.md",
            "--baseline-results",
            "results/counterfactual_2x2.csv",
        ],
        launcher=launcher,
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
            "results/counterfactual_2x2.csv",
            "--results-out",
            "results/aef_plus_results.csv",
            "--window-errors-out",
            "results/aef_plus_window_errors.csv",
            "--arg-out",
            "results/aef_plus_artifact_reliance_gap.csv",
            "--wgr-out",
            "results/aef_plus_worst_group_risk.csv",
            "--report-out",
            "reports/aef_plus_summary.md",
        ],
        launcher=launcher,
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
            "results/counterfactual_2x2.csv",
            "--results-out",
            "results/aif_plus_results.csv",
            "--window-errors-out",
            "results/aif_plus_window_errors.csv",
            "--arg-out",
            "results/aif_plus_artifact_reliance_gap.csv",
            "--wgr-out",
            "results/aif_plus_worst_group_risk.csv",
            "--ri-out",
            "results/aif_plus_ranking_instability.csv",
            "--report-out",
            "reports/aif_plus_summary.md",
        ],
        launcher=launcher,
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
        ],
        launcher=launcher,
    )
    run_cli(
        "build_unified_leaderboard_appendix",
        [
            "--results-dir",
            "results",
            "--reports-dir",
            "reports",
            "--support-summary",
            "reports/clean_view_support_summary.csv",
        ],
        launcher=launcher,
    )
    run_cli(
        "build_handoff_reports",
        [
            "--results-dir",
            "results",
            "--reports-dir",
            "reports",
            "--stats-dir",
            "statistic_results",
        ],
        launcher=launcher,
    )

    if not args.skip_organize:
        organize_args = ["--manifest-out", "cleanup_manifest.md"]
        if args.clean_handoff:
            organize_args.append("--clean-existing")
        run_cli("organize_handoff_outputs", organize_args, launcher=launcher)

    log_progress("pipeline finished")


if __name__ == "__main__":
    main()
