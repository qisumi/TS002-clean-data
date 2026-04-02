from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from data.io import ensure_project_directories, read_json, write_dataframe_csv, write_json, write_markdown
from data.paths import ROOT_DIR
from backbones.registry import discover_backbone_repo
from data.forecasting import load_dataset_bundle, load_events_lookup, load_view_frame
from experiments.selectors import RunSelector, select_specs
from experiments.spec import ExperimentSpec, RunContext, SettingResult
from reporting.counterfactual import (
    ARG_COLUMNS,
    RI_COLUMNS,
    WGR_COLUMNS,
    build_summary_markdown,
    compute_ri_table,
    compute_wgr_table,
    group_arg_rows,
    overall_arg_rows,
)
from training.evaluators import evaluate_forecaster
from training.logging import log_progress
from training.loops import TrainArtifacts, fit_forecaster
from views.selection import deterministic_subsample, resolve_validation_rows, select_view_rows
from utils.experiment_profiles import canonicalize_dataset_name, resolve_backbone_experiment


TERMINAL_STATUSES = {
    "completed",
    "skipped_missing_backbone_repo",
    "blocked_zero_strict_target_support",
    "blocked_no_view_support",
}


@dataclass(frozen=True)
class CounterfactualPaths:
    task_root: Path
    settings_root: Path
    result_table_path: Path
    window_errors_path: Path
    arg_path: Path
    wgr_path: Path
    ri_path: Path


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def default_counterfactual_manifest_path(task_root: Path) -> Path:
    return task_root / "manifests" / "counterfactual.jsonl"


def resolve_counterfactual_paths(task_root: Path) -> CounterfactualPaths:
    return CounterfactualPaths(
        task_root=task_root,
        settings_root=task_root / "settings",
        result_table_path=task_root / "counterfactual_2x2.csv",
        window_errors_path=task_root / "counterfactual_window_errors.csv",
        arg_path=task_root / "artifact_reliance_gap.csv",
        wgr_path=task_root / "worst_group_risk.csv",
        ri_path=task_root / "ranking_instability.csv",
    )


def resolve_c_view(config: dict[str, Any], dataset_name: str) -> str:
    return str(config["defaults"]["view_alias"][dataset_name]["clean_like"])


def sanitize_token(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or "NA"


def build_setting_id(setting_index: int, dataset_name: str, backbone: str, lookback: int, horizon: int, seed: int, train_view_name: str, eval_view_name: str, eval_protocol: str, subset_name: str) -> str:
    return (
        f"{setting_index:04d}_"
        f"{sanitize_token(dataset_name)}_"
        f"{sanitize_token(backbone)}_"
        f"L{int(lookback)}_"
        f"H{int(horizon)}_"
        f"seed{int(seed)}_"
        f"train-{sanitize_token(train_view_name)}_"
        f"eval-{sanitize_token(eval_view_name)}_"
        f"protocol-{sanitize_token(eval_protocol)}_"
        f"subset-{sanitize_token(subset_name)}"
    )


def build_counterfactual_specs(config: dict[str, Any]) -> list[ExperimentSpec]:
    defaults = config["defaults"]
    runtime_defaults = dict(defaults.get("runtime", {}))
    if defaults.get("lookback") is not None:
        runtime_defaults["lookback_override"] = int(defaults["lookback"])

    specs: list[ExperimentSpec] = []
    setting_index = 0
    for dataset_name_raw in defaults["datasets"]:
        for backbone_cfg in defaults["backbones"]:
            for horizon in defaults["horizons"]:
                for seed in defaults["seeds"]:
                    dataset_name = canonicalize_dataset_name(str(dataset_name_raw))
                    resolved = resolve_backbone_experiment(
                        backbone_cfg=dict(backbone_cfg),
                        dataset_name=dataset_name,
                        horizon=int(horizon),
                        runtime_defaults=runtime_defaults,
                    )
                    clean_view = resolve_c_view(config, dataset_name)
                    setting_rows: list[dict[str, Any]] = []
                    for train_token, eval_token in [("R", "R"), ("R", "C"), ("C", "R"), ("C", "C"), ("R", "I"), ("C", "I")]:
                        setting_rows.append(
                            {
                                "train_view_token": train_token,
                                "eval_view_token": eval_token,
                                "train_view_name": "raw" if train_token == "R" else clean_view,
                                "eval_view_name": "raw" if eval_token == "R" else clean_view if eval_token == "C" else "intervened",
                                "eval_row_view_name": "raw" if eval_token == "R" else clean_view if eval_token == "C" else "intervened",
                                "eval_protocol": "view_matrix",
                                "subset_name": "full_view",
                                "apply_eval_intervention": eval_token == "I",
                                "require_strict_target_clean": False,
                            }
                        )
                    if str(dataset_name) == "ETTh2":
                        for train_token, require_strict_target_clean in [("R", False), ("R", True), ("C", False), ("C", True)]:
                            subset_name = "target_clean_strict" if require_strict_target_clean else "overall_raw_test"
                            train_view_name = "raw" if train_token == "R" else clean_view
                            setting_rows.extend(
                                [
                                    {
                                        "train_view_token": train_token,
                                        "eval_view_token": "PR",
                                        "train_view_name": train_view_name,
                                        "eval_view_name": "raw_same_target",
                                        "eval_row_view_name": "raw",
                                        "eval_protocol": "paired_input_only",
                                        "subset_name": subset_name,
                                        "apply_eval_intervention": False,
                                        "require_strict_target_clean": require_strict_target_clean,
                                    },
                                    {
                                        "train_view_token": train_token,
                                        "eval_view_token": "PI",
                                        "train_view_name": train_view_name,
                                        "eval_view_name": "input_intervened_same_target",
                                        "eval_row_view_name": "raw",
                                        "eval_protocol": "paired_input_only",
                                        "subset_name": subset_name,
                                        "apply_eval_intervention": True,
                                        "require_strict_target_clean": require_strict_target_clean,
                                    },
                                ]
                            )

                    for row in setting_rows:
                        setting_index += 1
                        specs.append(
                            ExperimentSpec(
                                task_name="counterfactual_eval",
                                setting_id=build_setting_id(
                                    setting_index=setting_index,
                                    dataset_name=dataset_name,
                                    backbone=str(backbone_cfg["name"]),
                                    lookback=int(resolved.lookback),
                                    horizon=int(horizon),
                                    seed=int(seed),
                                    train_view_name=str(row["train_view_name"]),
                                    eval_view_name=str(row["eval_view_name"]),
                                    eval_protocol=str(row["eval_protocol"]),
                                    subset_name=str(row["subset_name"]),
                                ),
                                setting_index=setting_index,
                                dataset_name=dataset_name,
                                backbone=str(backbone_cfg["name"]),
                                lookback=int(resolved.lookback),
                                horizon=int(horizon),
                                seed=int(seed),
                                train_view_token=str(row["train_view_token"]),
                                eval_view_token=str(row["eval_view_token"]),
                                train_view_name=str(row["train_view_name"]),
                                eval_view_name=str(row["eval_view_name"]),
                                eval_row_view_name=str(row["eval_row_view_name"]),
                                eval_protocol=str(row["eval_protocol"]),
                                subset_name=str(row["subset_name"]),
                                apply_eval_intervention=bool(row["apply_eval_intervention"]),
                                runtime_cfg=dict(resolved.runtime),
                                model_params=dict(resolved.model_params),
                                source_meta={
                                    "hyperparam_source_url": str(resolved.source_url),
                                    "hyperparam_source_kind": str(resolved.source_kind),
                                    "hyperparam_source_note": str(resolved.source_note),
                                },
                                extra={
                                    "require_strict_target_clean": bool(row["require_strict_target_clean"]),
                                },
                            )
                        )
    return specs


def summarize_backbone_status(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    statuses: dict[str, dict[str, str]] = {}
    for backbone_cfg in config["defaults"]["backbones"]:
        repo = discover_backbone_repo(ROOT_DIR, list(backbone_cfg.get("repo_candidates", [])))
        statuses[str(backbone_cfg["name"])] = {
            "status": "available" if repo is not None else "missing",
            "repo_path": str(repo) if repo is not None else "",
        }
    return statuses


def annotate_subset_flags(rows: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    out = rows.copy()
    out["has_input_intervention"] = (
        out["n_events_input"].fillna(0).astype(float).gt(0)
        | out["intervention_recipe"].fillna("").astype(str).ne("")
    ).astype(int)
    out["strict_target_clean"] = out["n_events_target"].fillna(0).astype(float).eq(0).astype(int)
    out["subset_name"] = subset_name
    return out


def resolve_eval_rows_for_spec(view_df: pd.DataFrame, spec: ExperimentSpec, max_rows: int | None) -> tuple[pd.DataFrame, int]:
    if spec.eval_protocol == "paired_input_only":
        source_rows = select_view_rows(view_df, split_name="test", view_name=spec.eval_row_view_name, max_rows=None)
        source_rows = annotate_subset_flags(source_rows, subset_name=spec.subset_name)
        source_support = int(len(source_rows))
        filtered = source_rows
        if bool(spec.extra.get("require_strict_target_clean", False)):
            filtered = filtered[filtered["strict_target_clean"] == 1].copy()
        return deterministic_subsample(filtered, max_rows=max_rows), source_support

    eval_rows = select_view_rows(view_df, split_name="test", view_name=spec.eval_row_view_name, max_rows=max_rows)
    eval_rows = annotate_subset_flags(eval_rows, subset_name=spec.subset_name)
    return eval_rows, int(len(eval_rows))


def build_setting_log_path(setting_logs_dir: Path, spec: ExperimentSpec) -> Path:
    return setting_logs_dir / f"{spec.setting_id}.log"


def initialize_setting_log(setting_log_path: Path, spec: ExperimentSpec) -> None:
    header_lines = [
        "# Counterfactual Eval Setting Log",
        f"setting_index={spec.setting_index}",
        f"setting_id={spec.setting_id}",
        f"dataset_name={spec.dataset_name}",
        f"backbone={spec.backbone}",
        f"lookback={spec.lookback}",
        f"horizon={spec.horizon}",
        f"seed={spec.seed}",
        f"train_view_name={spec.train_view_name}",
        f"eval_view_name={spec.eval_view_name}",
        f"eval_protocol={spec.eval_protocol}",
        f"subset_name={spec.subset_name}",
        f"hyperparam_source_kind={spec.source_meta.get('hyperparam_source_kind', '')}",
        f"hyperparam_source_url={spec.source_meta.get('hyperparam_source_url', '')}",
        f"hyperparam_source_note={spec.source_meta.get('hyperparam_source_note', '')}",
        "",
    ]
    if not setting_log_path.exists():
        setting_log_path.parent.mkdir(parents=True, exist_ok=True)
        setting_log_path.write_text("\n".join(header_lines), encoding="utf-8")
        return

    with setting_log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write(f"# Resume Session {datetime.now().isoformat(timespec='seconds')}\n")
        handle.write(f"setting_index={spec.setting_index}\n")
        handle.write(f"setting_id={spec.setting_id}\n")
        handle.write("\n")


def setting_output_dir(task_root: Path, setting_id: str) -> Path:
    return task_root / "settings" / setting_id


def result_json_path(task_root: Path, setting_id: str) -> Path:
    return setting_output_dir(task_root, setting_id) / "result.json"


def window_errors_output_path(task_root: Path, setting_id: str) -> Path:
    return setting_output_dir(task_root, setting_id) / "window_errors.csv"


def load_existing_setting_result(path: Path) -> SettingResult | None:
    if not path.exists():
        return None
    try:
        return SettingResult.from_dict(read_json(path))
    except Exception:
        return None


def normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def finalize_setting_result(spec: ExperimentSpec, result_row: dict[str, Any], metrics: dict[str, Any], artifact_paths: dict[str, str], error_rows_path: str) -> SettingResult:
    normalized_row = {key: normalize_value(value) for key, value in result_row.items()}
    normalized_metrics = {key: normalize_value(value) for key, value in metrics.items()}
    return SettingResult(
        task_name=spec.task_name,
        setting_id=spec.setting_id,
        setting_index=spec.setting_index,
        status=str(normalized_row["status"]),
        metrics=normalized_metrics,
        n_train_windows=int(normalized_row.get("n_train_windows") or 0),
        n_eval_windows=int(normalized_row.get("n_eval_windows") or 0),
        best_val_metric=normalized_row.get("best_val_mse", normalized_row.get("best_val_mae")),
        epochs_ran=normalized_row.get("epochs_ran"),
        fit_seconds=normalized_row.get("fit_seconds"),
        artifact_paths={key: str(value) for key, value in artifact_paths.items()},
        error_rows_path=error_rows_path,
        result_row=normalized_row,
        spec=spec.to_dict(),
    )


def write_setting_outputs(task_root: Path, setting_result: SettingResult, window_errors: pd.DataFrame | None) -> None:
    output_dir = setting_output_dir(task_root, setting_result.setting_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    if window_errors is not None:
        write_dataframe_csv(window_errors_output_path(task_root, setting_result.setting_id), window_errors, index=False)
    write_json(result_json_path(task_root, setting_result.setting_id), setting_result.to_dict())


def run_one_setting(
    spec: ExperimentSpec,
    context: RunContext,
    backbone_status: dict[str, dict[str, str]],
    dataset_cache: dict[str, Any],
    view_cache: dict[tuple[str, int, int], pd.DataFrame],
    events_cache: dict[str, dict[str, Any]],
    model_cache: dict[tuple[str, str, int, int, int, str], TrainArtifacts],
    train_log_cache: dict[tuple[str, str, int, int, int, str], str],
) -> SettingResult:
    task_root = context.task_root
    prior_result = load_existing_setting_result(result_json_path(task_root, spec.setting_id))
    if prior_result is not None and prior_result.status in TERMINAL_STATUSES:
        log_progress(f"[006] skip existing setting result {spec.setting_index} setting_id={spec.setting_id}")
        return prior_result

    setting_log_path = build_setting_log_path(context.setting_logs_dir, spec)
    initialize_setting_log(setting_log_path, spec)
    log_progress(
        f"[006] setting {spec.setting_index} setting_id={spec.setting_id} dataset={spec.dataset_name} backbone={spec.backbone} "
        f"L{spec.lookback} H{spec.horizon} train={spec.train_view_name} eval={spec.eval_view_name} "
        f"protocol={spec.eval_protocol} subset={spec.subset_name}",
        log_path=setting_log_path,
    )

    bb_state = backbone_status[spec.backbone]
    view_key = (spec.dataset_name, spec.lookback, spec.horizon)
    if view_key not in view_cache:
        view_cache[view_key] = load_view_frame(context.views_dir, spec.dataset_name, lookback=spec.lookback, horizon=spec.horizon)
    current_view = view_cache[view_key]
    current_runtime = dict(spec.runtime_cfg)
    train_rows = select_view_rows(
        current_view,
        split_name="train",
        view_name=spec.train_view_name,
        max_rows=current_runtime.get("max_train_windows"),
    )
    eval_rows, eval_source_support = resolve_eval_rows_for_spec(
        current_view,
        spec=spec,
        max_rows=current_runtime.get("max_test_windows"),
    )
    train_support = int(len(train_rows))
    eval_support = int(len(eval_rows))

    row_common = {
        "setting_index": spec.setting_index,
        "setting_id": spec.setting_id,
        "dataset_name": spec.dataset_name,
        "backbone": spec.backbone,
        "lookback": spec.lookback,
        "horizon": spec.horizon,
        "seed": spec.seed,
        "train_view_token": spec.train_view_token,
        "eval_view_token": spec.eval_view_token,
        "train_view_name": spec.train_view_name,
        "eval_view_name": spec.eval_view_name,
        "eval_row_view_name": spec.eval_row_view_name,
        "eval_protocol": spec.eval_protocol,
        "subset_name": spec.subset_name,
        "n_train_windows": train_support,
        "n_eval_windows": eval_support,
        "n_eval_source_windows": eval_source_support,
        "repo_status": bb_state["status"],
        "repo_path": bb_state["repo_path"],
        "hyperparam_source_kind": str(spec.source_meta.get("hyperparam_source_kind", "")),
        "hyperparam_source_url": str(spec.source_meta.get("hyperparam_source_url", "")),
        "hyperparam_source_note": str(spec.source_meta.get("hyperparam_source_note", "")),
        "setting_log_path": str(setting_log_path),
    }
    artifact_paths = {
        "result_json": str(result_json_path(task_root, spec.setting_id)),
        "window_errors_csv": str(window_errors_output_path(task_root, spec.setting_id)),
        "setting_log": str(setting_log_path),
    }

    if bb_state["status"] == "missing":
        log_progress(f"[006] skip missing backbone repo: {spec.backbone}", log_path=setting_log_path)
        result = finalize_setting_result(
            spec=spec,
            result_row={
                **row_common,
                "status": "skipped_missing_backbone_repo",
                "skip_reason": f"{spec.backbone} repo not found",
                "val_view_name": "",
                "best_val_mae": pd.NA,
                "best_val_mse": pd.NA,
                "epochs_ran": pd.NA,
                "fit_seconds": pd.NA,
                "mae": pd.NA,
                "mse": pd.NA,
                "smape": pd.NA,
            },
            metrics={},
            artifact_paths=artifact_paths,
            error_rows_path="",
        )
        write_setting_outputs(task_root, result, window_errors=None)
        return result

    if spec.eval_protocol == "paired_input_only" and bool(spec.extra.get("require_strict_target_clean", False)) and eval_support == 0:
        log_progress(
            f"[006] blocked_zero_strict_target_support dataset={spec.dataset_name} H{spec.horizon} "
            f"train={spec.train_view_name}",
            log_path=setting_log_path,
        )
        result = finalize_setting_result(
            spec=spec,
            result_row={
                **row_common,
                "status": "blocked_zero_strict_target_support",
                "skip_reason": "no raw test windows satisfy n_events_target == 0",
                "val_view_name": "",
                "best_val_mae": pd.NA,
                "best_val_mse": pd.NA,
                "epochs_ran": pd.NA,
                "fit_seconds": pd.NA,
                "mae": pd.NA,
                "mse": pd.NA,
                "smape": pd.NA,
            },
            metrics={},
            artifact_paths=artifact_paths,
            error_rows_path="",
        )
        write_setting_outputs(task_root, result, window_errors=None)
        return result

    if train_support == 0 or eval_support == 0:
        log_progress(
            f"[006] blocked_no_view_support train_support={train_support} eval_support={eval_support}",
            log_path=setting_log_path,
        )
        result = finalize_setting_result(
            spec=spec,
            result_row={
                **row_common,
                "status": "blocked_no_view_support",
                "skip_reason": "missing train/eval view support",
                "val_view_name": "",
                "best_val_mae": pd.NA,
                "best_val_mse": pd.NA,
                "epochs_ran": pd.NA,
                "fit_seconds": pd.NA,
                "mae": pd.NA,
                "mse": pd.NA,
                "smape": pd.NA,
            },
            metrics={},
            artifact_paths=artifact_paths,
            error_rows_path="",
        )
        write_setting_outputs(task_root, result, window_errors=None)
        return result

    if spec.dataset_name not in dataset_cache:
        log_progress(f"[006] loading dataset bundle for {spec.dataset_name}", log_path=setting_log_path)
        dataset_cache[spec.dataset_name] = load_dataset_bundle(spec.dataset_name, registry_path=context.registry_path)
        events_cache[spec.dataset_name] = load_events_lookup(events_path=context.events_path, dataset_name=spec.dataset_name)

    train_key = (
        spec.dataset_name,
        spec.backbone,
        spec.lookback,
        spec.horizon,
        spec.seed,
        spec.train_view_name,
    )
    if train_key not in model_cache:
        val_view_name, val_rows = resolve_validation_rows(
            current_view,
            train_view_name=spec.train_view_name,
            max_val_rows=current_runtime.get("max_val_windows"),
        )
        if train_rows.empty or val_rows.empty:
            log_progress(
                f"[006] blocked train/val rows unavailable dataset={spec.dataset_name} H{spec.horizon} "
                f"train={spec.train_view_name}",
                log_path=setting_log_path,
            )
            result = finalize_setting_result(
                spec=spec,
                result_row={
                    **row_common,
                    "status": "blocked_no_view_support",
                    "skip_reason": "train/val rows unavailable after adapter selection",
                    "val_view_name": val_view_name,
                    "best_val_mae": pd.NA,
                    "best_val_mse": pd.NA,
                    "epochs_ran": pd.NA,
                    "fit_seconds": pd.NA,
                    "mae": pd.NA,
                    "mse": pd.NA,
                    "smape": pd.NA,
                },
                metrics={},
                artifact_paths=artifact_paths,
                error_rows_path="",
            )
            write_setting_outputs(task_root, result, window_errors=None)
            return result

        model_cache[train_key] = fit_forecaster(
            backbone_name=spec.backbone,
            model_params=dict(spec.model_params),
            runtime_cfg=current_runtime,
            dataset_bundle=dataset_cache[spec.dataset_name],
            events_lookup=events_cache[spec.dataset_name],
            train_rows=train_rows,
            val_rows=val_rows,
            train_view_name=spec.train_view_name,
            val_view_name=val_view_name,
            seed=spec.seed,
            log_prefix=f"[006] {spec.dataset_name}/{spec.backbone}/L{spec.lookback}/H{spec.horizon}/train={spec.train_view_name}",
            log_path=setting_log_path,
        )
        train_log_cache[train_key] = str(setting_log_path)
    else:
        log_progress(
            f"[006] reusing cached model dataset={spec.dataset_name} backbone={spec.backbone} "
            f"L{spec.lookback} H{spec.horizon} train={spec.train_view_name}; "
            f"trained_in={train_log_cache.get(train_key, 'unknown')}",
            log_path=setting_log_path,
        )

    artifacts = model_cache[train_key]
    metrics, window_errors = evaluate_forecaster(
        model=artifacts.model,
        dataset_bundle=dataset_cache[spec.dataset_name],
        events_lookup=events_cache[spec.dataset_name],
        eval_rows=eval_rows,
        runtime_cfg=current_runtime,
        apply_intervention=spec.apply_eval_intervention,
        setting_meta={
            "setting_index": spec.setting_index,
            "setting_id": spec.setting_id,
            "dataset_name": spec.dataset_name,
            "backbone": spec.backbone,
            "lookback": spec.lookback,
            "horizon": spec.horizon,
            "seed": spec.seed,
            "train_view_name": spec.train_view_name,
            "train_view_token": spec.train_view_token,
            "eval_view_name": spec.eval_view_name,
            "eval_view_token": spec.eval_view_token,
            "eval_protocol": spec.eval_protocol,
            "subset_name": spec.subset_name,
        },
        log_path=setting_log_path,
        log_prefix=f"[006] {spec.dataset_name}/{spec.backbone}/L{spec.lookback}/H{spec.horizon}/eval={spec.eval_view_name}",
    )
    log_progress(
        f"[006] eval done dataset={spec.dataset_name} backbone={spec.backbone} "
        f"L{spec.lookback} H{spec.horizon} eval={spec.eval_view_name} subset={spec.subset_name} "
        f"mae={float(metrics['mae']):.6f}",
        log_path=setting_log_path,
    )
    result = finalize_setting_result(
        spec=spec,
        result_row={
            **row_common,
            "status": "completed",
            "skip_reason": "",
            "val_view_name": artifacts.val_view_name,
            "best_val_mae": round(float(artifacts.best_val_mae), 6),
            "best_val_mse": round(float(artifacts.best_val_mse), 6),
            "epochs_ran": int(artifacts.epochs_ran),
            "fit_seconds": float(artifacts.fit_seconds),
            "mae": round(float(metrics["mae"]), 6),
            "mse": round(float(metrics["mse"]), 6),
            "smape": round(float(metrics["smape"]), 6),
        },
        metrics={
            "mae": round(float(metrics["mae"]), 6),
            "mse": round(float(metrics["mse"]), 6),
            "smape": round(float(metrics["smape"]), 6),
        },
        artifact_paths=artifact_paths,
        error_rows_path=artifact_paths["window_errors_csv"],
    )
    write_setting_outputs(task_root, result, window_errors=window_errors)
    return result


def collect_setting_results(task_root: Path) -> list[SettingResult]:
    settings_root = resolve_counterfactual_paths(task_root).settings_root
    if not settings_root.exists():
        return []
    results: list[SettingResult] = []
    for result_path in sorted(settings_root.glob("*/result.json")):
        try:
            results.append(SettingResult.from_dict(read_json(result_path)))
        except Exception:
            continue
    return sorted(results, key=lambda item: item.setting_index)


def merge_counterfactual_outputs(task_root: Path, report_out: Path, setting_logs_dir: Path, backbone_status: dict[str, dict[str, str]]) -> dict[str, pd.DataFrame]:
    ensure_project_directories()
    paths = resolve_counterfactual_paths(task_root)
    setting_results = collect_setting_results(task_root)
    result_rows = [item.result_row for item in setting_results]
    results_df = pd.DataFrame(result_rows).sort_values("setting_index") if result_rows else pd.DataFrame()
    if not results_df.empty:
        write_dataframe_csv(paths.result_table_path, results_df, index=False)
    else:
        write_dataframe_csv(paths.result_table_path, pd.DataFrame(), index=False)

    window_error_frames: list[pd.DataFrame] = []
    for item in setting_results:
        csv_path = Path(item.error_rows_path) if item.error_rows_path else window_errors_output_path(task_root, item.setting_id)
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, low_memory=False)
        if not frame.empty:
            window_error_frames.append(frame)
    window_errors_df = pd.concat(window_error_frames, ignore_index=True) if window_error_frames else pd.DataFrame()
    write_dataframe_csv(paths.window_errors_path, window_errors_df, index=False)

    arg_rows = overall_arg_rows(results_df) if not results_df.empty else []
    if not window_errors_df.empty:
        arg_rows.extend(group_arg_rows(window_errors_df))
    arg_df = pd.DataFrame(arg_rows, columns=ARG_COLUMNS)
    write_dataframe_csv(paths.arg_path, arg_df, index=False)

    wgr_df = compute_wgr_table(window_errors_df) if not window_errors_df.empty else pd.DataFrame(columns=WGR_COLUMNS)
    write_dataframe_csv(paths.wgr_path, wgr_df, index=False)

    ri_df = compute_ri_table(results_df) if not results_df.empty else pd.DataFrame(columns=RI_COLUMNS)
    write_dataframe_csv(paths.ri_path, ri_df, index=False)

    write_markdown(
        report_out,
        build_summary_markdown(
            backbone_status=backbone_status,
            results_df=results_df if not results_df.empty else pd.DataFrame(columns=["status", "mae"]),
            arg_df=arg_df,
            wgr_df=wgr_df,
            ri_df=ri_df,
            setting_logs_dir=setting_logs_dir,
        ),
    )
    log_progress(
        f"[006] merge finished results={paths.result_table_path} completed="
        f"{int((results_df['status'] == 'completed').sum()) if not results_df.empty and 'status' in results_df.columns else 0}"
    )
    return {
        "results_df": results_df,
        "window_errors_df": window_errors_df,
        "arg_df": arg_df,
        "wgr_df": wgr_df,
        "ri_df": ri_df,
    }
def run_counterfactual_specs(
    specs: list[ExperimentSpec],
    selector: RunSelector,
    context: RunContext,
    auto_merge: bool,
) -> list[SettingResult]:
    ensure_project_directories()
    context.task_root.mkdir(parents=True, exist_ok=True)
    context.report_out.parent.mkdir(parents=True, exist_ok=True)
    context.setting_logs_dir.mkdir(parents=True, exist_ok=True)

    selected_specs = select_specs(specs, selector)
    backbone_status = summarize_backbone_status(load_config(context.config_path))
    log_progress(
        f"[006] run_counterfactual_eval start settings={len(specs)} selected={len(selected_specs)} "
        f"task_root={context.task_root}"
    )

    dataset_cache: dict[str, Any] = {}
    view_cache: dict[tuple[str, int, int], pd.DataFrame] = {}
    events_cache: dict[str, dict[str, Any]] = {}
    model_cache: dict[tuple[str, str, int, int, int, str], TrainArtifacts] = {}
    train_log_cache: dict[tuple[str, str, int, int, int, str], str] = {}
    setting_results: list[SettingResult] = []
    for spec in selected_specs:
        setting_results.append(
            run_one_setting(
                spec=spec,
                context=context,
                backbone_status=backbone_status,
                dataset_cache=dataset_cache,
                view_cache=view_cache,
                events_cache=events_cache,
                model_cache=model_cache,
                train_log_cache=train_log_cache,
            )
        )

    if auto_merge:
        merge_counterfactual_outputs(
            task_root=context.task_root,
            report_out=context.report_out,
            setting_logs_dir=context.setting_logs_dir,
            backbone_status=backbone_status,
        )
    return setting_results
