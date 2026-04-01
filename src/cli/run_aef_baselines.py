from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data import write_markdown
from experiments.aef_shared import (
    AEFRegressor,
    TabularFeatureEncoder,
    TabularForecastDataset,
    build_feature_frame,
    build_loader,
    build_targets,
    compare_with_standard,
    control_feature_mask,
    evaluate_aef_model,
    fit_aef_model,
    load_config,
    shuffle_control_features,
)
from utils.forecasting_utils import (
    load_dataset_bundle,
    load_view_frame,
    resolve_device,
    resolve_validation_rows,
    select_view_rows,
)


def log_progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [007/AEF] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AEF-Weak and AEF-Control minimal diagnostics.")
    parser.add_argument("--config", default=str(Path("configs") / "aef.yaml"))
    parser.add_argument("--views-dir", default=str(Path("statistic_results") / "window_views"))
    parser.add_argument("--registry", default=str(Path("statistic_results") / "dataset_registry.csv"))
    parser.add_argument("--results-dir", default=str(Path("results")))
    parser.add_argument("--report-out", default=str(Path("reports") / "aef_summary.md"))
    parser.add_argument("--baseline-results", default=str(Path("results") / "counterfactual_2x2.csv"))
    return parser.parse_args()


def build_summary_markdown(
    weak_df: pd.DataFrame,
    control_df: pd.DataFrame,
    baseline_results: pd.DataFrame,
) -> str:
    weak_cmp = compare_with_standard(weak_df, baseline_results)
    control_cmp = compare_with_standard(control_df, baseline_results)
    merged = weak_cmp.merge(
        control_cmp,
        on=["dataset_name", "horizon", "eval_view_name"],
        suffixes=("_weak", "_control"),
    )

    lines = [
        "# AEF Summary",
        "",
        "本轮把 AEF 落成了 metadata-aware 的诊断器，而不是新的主干 forecaster。",
        "",
        "## 结果概览",
        "",
        f"- AEF-Weak setting 数: {len(weak_df)}",
        f"- AEF-Control setting 数: {len(control_df)}",
        "",
        "## Falsification Signals",
        "",
    ]

    if not merged.empty:
        merged["weak_minus_control_raw"] = np.where(
            merged["eval_view_name"] == "raw",
            merged["mae_weak"] - merged["mae_control"],
            np.nan,
        )
        merged["weak_intervention_gap"] = merged["mae_weak"] - merged["best_forecaster_mae_weak"]
        raw_rows = merged[merged["eval_view_name"] == "raw"].sort_values("mae_control", ascending=False)
        for row in raw_rows.head(6).itertuples(index=False):
            lines.append(
                f"- raw {row.dataset_name} / H{row.horizon}: Weak={row.mae_weak:.4f}, Control={row.mae_control:.4f}, "
                f"best_forecaster={row.best_forecaster_mae_weak:.4f}"
            )

        for dataset_name, horizon in sorted(set(zip(merged["dataset_name"], merged["horizon"]))):
            ds_rows = merged[(merged["dataset_name"] == dataset_name) & (merged["horizon"] == horizon)]
            raw_weak = ds_rows.loc[ds_rows["eval_view_name"] == "raw", "mae_weak"]
            int_weak = ds_rows.loc[ds_rows["eval_view_name"] == "intervened", "mae_weak"]
            raw_control = ds_rows.loc[ds_rows["eval_view_name"] == "raw", "mae_control"]
            int_control = ds_rows.loc[ds_rows["eval_view_name"] == "intervened", "mae_control"]
            if raw_weak.empty or int_weak.empty or raw_control.empty or int_control.empty:
                continue
            weak_arg = float((int_weak.iloc[0] - raw_weak.iloc[0]) / max(raw_weak.iloc[0], 1e-8))
            control_arg = float((int_control.iloc[0] - raw_control.iloc[0]) / max(raw_control.iloc[0], 1e-8))
            lines.append(
                f"- {dataset_name} / H{horizon}: Weak_ARG={weak_arg:.4f}, Control_ARG={control_arg:.4f}"
            )
    else:
        lines.append("- 当前尚无可对比的 AEF 结果。")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 若 `AEF-Weak(raw)` 明显优于 `AEF-Control(raw)`，且 `Weak_ARG > Control_ARG`，则说明 raw benchmark 中存在可被 metadata-aware 规则利用的 shortcut gain。",
            "- 若 Weak 与 Control 差异很小，优先回查 feature 设计和 006 的 view 语义，而不是直接推进 `AEF-Plus/AIF-Plus`。",
            "",
        ]
    )
    return "\n".join(lines)


def run_aef(config: dict[str, Any], args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    lookback = int(config["defaults"].get("lookback", 96))
    views_dir = Path(args.views_dir)
    registry_path = Path(args.registry)
    runtime_cfg = dict(config["defaults"].get("runtime", {}))
    model_cfg = dict(config["defaults"].get("model", {}))
    feature_cfg = dict(config["defaults"].get("feature", {}))

    feature_cache: dict[tuple[str, int, str, str], tuple[pd.DataFrame, pd.DataFrame, np.ndarray]] = {}
    bundle_cache: dict[str, Any] = {}
    view_cache: dict[tuple[str, int], pd.DataFrame] = {}

    weak_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    weak_error_frames: list[pd.DataFrame] = []
    control_error_frames: list[pd.DataFrame] = []
    log_progress(
        f"start datasets={config['defaults']['datasets']} horizons={config['defaults']['horizons']} "
        f"train_view={config['defaults']['train_view_name']}"
    )

    for dataset_name in config["defaults"]["datasets"]:
        if dataset_name not in bundle_cache:
            bundle_cache[dataset_name] = load_dataset_bundle(dataset_name, registry_path=registry_path)
        for horizon in config["defaults"]["horizons"]:
            log_progress(f"prepare dataset={dataset_name} horizon={horizon}")
            view_key = (dataset_name, int(horizon))
            if view_key not in view_cache:
                view_cache[view_key] = load_view_frame(views_dir, dataset_name, lookback=lookback, horizon=int(horizon))

            bundle = bundle_cache[dataset_name]
            view_df = view_cache[view_key]
            train_rows = select_view_rows(
                view_df,
                split_name="train",
                view_name=str(config["defaults"]["train_view_name"]),
                max_rows=runtime_cfg.get("max_train_windows"),
            )
            val_view_name, val_rows = resolve_validation_rows(
                view_df,
                train_view_name=str(config["defaults"]["train_view_name"]),
                max_val_rows=runtime_cfg.get("max_val_windows"),
            )
            if train_rows.empty or val_rows.empty:
                log_progress(f"skip dataset={dataset_name} horizon={horizon} due to empty train/val rows")
                continue

            train_features_df = build_feature_frame(train_rows, bundle=bundle, feature_cfg=feature_cfg)
            val_features_df = build_feature_frame(val_rows, bundle=bundle, feature_cfg=feature_cfg)
            train_targets = build_targets(train_rows, bundle=bundle)
            val_targets = build_targets(val_rows, bundle=bundle)

            # Auto-detect categorical columns so new metadata fields do not silently
            # fall through into the numeric block.
            categorical_cols = sorted(col for col in train_features_df.columns if col.startswith("cat_"))
            encoder = TabularFeatureEncoder(categorical_cols=categorical_cols)
            encoder.fit(train_features_df)
            train_x = encoder.transform(train_features_df)
            val_x = encoder.transform(val_features_df)
            control_mask = control_feature_mask(encoder.feature_names)

            output_dim = int(train_targets.shape[1])
            device = resolve_device(str(runtime_cfg.get("device", "auto")))
            amp_enabled = bool(runtime_cfg.get("amp", True)) and device.type == "cuda"
            pin_memory = bool(runtime_cfg.get("pin_memory", True)) and device.type == "cuda"

            weak_model = AEFRegressor(
                input_dim=int(train_x.shape[1]),
                output_dim=output_dim,
                hidden_dim=int(model_cfg.get("hidden_dim", 128)),
                hidden_layers=int(model_cfg.get("hidden_layers", 2)),
                dropout=float(model_cfg.get("dropout", 0.1)),
            )
            weak_model, weak_best_val_mae, weak_epochs = fit_aef_model(
                weak_model,
                TabularForecastDataset(train_x, train_targets, train_rows),
                TabularForecastDataset(val_x, val_targets, val_rows),
                runtime_cfg=runtime_cfg,
                seed=int(config["defaults"]["seeds"][0]),
                log_prefix=f"{dataset_name}/H{horizon}/AEF-Weak",
                log_fn=log_progress,
            )
            weak_model.to(device)

            control_train_x = shuffle_control_features(train_x, control_mask, seed=17 + int(horizon))
            control_val_x = shuffle_control_features(val_x, control_mask, seed=97 + int(horizon))
            control_model = AEFRegressor(
                input_dim=int(train_x.shape[1]),
                output_dim=output_dim,
                hidden_dim=int(model_cfg.get("hidden_dim", 128)),
                hidden_layers=int(model_cfg.get("hidden_layers", 2)),
                dropout=float(model_cfg.get("dropout", 0.1)),
            )
            control_model, control_best_val_mae, control_epochs = fit_aef_model(
                control_model,
                TabularForecastDataset(control_train_x, train_targets, train_rows),
                TabularForecastDataset(control_val_x, val_targets, val_rows),
                runtime_cfg=runtime_cfg,
                seed=101 + int(config["defaults"]["seeds"][0]),
                log_prefix=f"{dataset_name}/H{horizon}/AEF-Control",
                log_fn=log_progress,
            )
            control_model.to(device)

            for eval_view_name in config["defaults"]["eval_views"][dataset_name]:
                log_progress(f"evaluate dataset={dataset_name} horizon={horizon} view={eval_view_name}")
                cache_key = (dataset_name, int(horizon), "test", str(eval_view_name))
                if cache_key not in feature_cache:
                    eval_rows = select_view_rows(
                        view_df,
                        split_name="test",
                        view_name=str(eval_view_name),
                        max_rows=runtime_cfg.get("max_test_windows"),
                    )
                    feature_cache[cache_key] = (
                        eval_rows,
                        build_feature_frame(eval_rows, bundle=bundle, feature_cfg=feature_cfg),
                        build_targets(eval_rows, bundle=bundle),
                    )

                eval_rows, eval_features_df, eval_targets = feature_cache[cache_key]
                if eval_rows.empty:
                    continue
                eval_x = encoder.transform(eval_features_df)
                eval_loader = build_loader(
                    TabularForecastDataset(eval_x, eval_targets, eval_rows),
                    batch_size=int(runtime_cfg.get("eval_batch_size", 256)),
                    shuffle=False,
                    num_workers=int(runtime_cfg.get("num_workers", 0)),
                    pin_memory=pin_memory,
                )
                weak_metrics, weak_errors = evaluate_aef_model(
                    weak_model,
                    eval_loader,
                    device=device,
                    amp_enabled=amp_enabled,
                    meta={
                        "dataset_name": dataset_name,
                        "lookback": int(lookback),
                        "horizon": int(horizon),
                        "train_view_name": str(config["defaults"]["train_view_name"]),
                        "eval_view_name": str(eval_view_name),
                        "model_name": "AEF-Weak",
                    },
                )
                weak_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "lookback": int(lookback),
                        "horizon": int(horizon),
                        "train_view_name": str(config["defaults"]["train_view_name"]),
                        "eval_view_name": str(eval_view_name),
                        "model_name": "AEF-Weak",
                        "n_train_windows": int(len(train_rows)),
                        "n_eval_windows": int(len(eval_rows)),
                        "best_val_mae": round(float(weak_best_val_mae), 6),
                        "epochs_ran": int(weak_epochs),
                        "mae": round(float(weak_metrics["mae"]), 6),
                        "mse": round(float(weak_metrics["mse"]), 6),
                        "smape": round(float(weak_metrics["smape"]), 6),
                    }
                )
                weak_error_frames.append(weak_errors)
                log_progress(
                    f"done weak dataset={dataset_name} H{horizon} view={eval_view_name} "
                    f"mae={float(weak_metrics['mae']):.6f}"
                )

                control_eval_x = shuffle_control_features(eval_x, control_mask, seed=197 + int(horizon) + len(eval_rows))
                control_loader = build_loader(
                    TabularForecastDataset(control_eval_x, eval_targets, eval_rows),
                    batch_size=int(runtime_cfg.get("eval_batch_size", 256)),
                    shuffle=False,
                    num_workers=int(runtime_cfg.get("num_workers", 0)),
                    pin_memory=pin_memory,
                )
                control_metrics, control_errors = evaluate_aef_model(
                    control_model,
                    control_loader,
                    device=device,
                    amp_enabled=amp_enabled,
                    meta={
                        "dataset_name": dataset_name,
                        "lookback": int(lookback),
                        "horizon": int(horizon),
                        "train_view_name": str(config["defaults"]["train_view_name"]),
                        "eval_view_name": str(eval_view_name),
                        "model_name": "AEF-Control",
                    },
                )
                control_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "lookback": int(lookback),
                        "horizon": int(horizon),
                        "train_view_name": str(config["defaults"]["train_view_name"]),
                        "eval_view_name": str(eval_view_name),
                        "model_name": "AEF-Control",
                        "n_train_windows": int(len(train_rows)),
                        "n_eval_windows": int(len(eval_rows)),
                        "best_val_mae": round(float(control_best_val_mae), 6),
                        "epochs_ran": int(control_epochs),
                        "mae": round(float(control_metrics["mae"]), 6),
                        "mse": round(float(control_metrics["mse"]), 6),
                        "smape": round(float(control_metrics["smape"]), 6),
                    }
                )
                control_error_frames.append(control_errors)
                log_progress(
                    f"done control dataset={dataset_name} H{horizon} view={eval_view_name} "
                    f"mae={float(control_metrics['mae']):.6f}"
                )

    weak_df = pd.DataFrame(weak_rows)
    control_df = pd.DataFrame(control_rows)
    weak_df.to_csv(results_dir / "aef_results.csv", index=False)
    control_df.to_csv(results_dir / "aef_control_results.csv", index=False)
    (pd.concat(weak_error_frames, ignore_index=True) if weak_error_frames else pd.DataFrame()).to_csv(
        results_dir / "aef_window_errors.csv",
        index=False,
    )
    (pd.concat(control_error_frames, ignore_index=True) if control_error_frames else pd.DataFrame()).to_csv(
        results_dir / "aef_control_window_errors.csv",
        index=False,
    )

    baseline_results = pd.read_csv(args.baseline_results) if Path(args.baseline_results).exists() else pd.DataFrame()
    write_markdown(Path(args.report_out), build_summary_markdown(weak_df, control_df, baseline_results))
    log_progress(f"finished results_dir={results_dir} weak_rows={len(weak_df)} control_rows={len(control_df)}")


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    run_aef(config, args)


if __name__ == "__main__":
    main()
