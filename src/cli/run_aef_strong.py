from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd

from data import ROOT_DIR, write_markdown
from experiments.aef_shared import (
    TabularFeatureEncoder,
    TabularForecastDataset,
    artifact_phase_cols,
    blocked_permute_frame,
    build_feature_frame,
    build_loader,
    build_targets,
    compare_with_standard,
    evaluate_aef_model,
    fit_aef_variant,
    load_config,
    series_only_cols,
)
from utils.forecasting_utils import (
    load_dataset_bundle,
    load_view_frame,
    resolve_device,
    resolve_validation_rows,
    select_view_rows,
)


def log_progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [011/AEF-Strong] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AEF-Strong / ControlBlocked / SeriesOnly diagnostics.")
    parser.add_argument("--config", default=str(Path("configs") / "aef.yaml"))
    parser.add_argument("--views-dir", default=str(Path("statistic_results") / "window_views"))
    parser.add_argument("--registry", default=str(Path("statistic_results") / "dataset_registry.csv"))
    parser.add_argument("--results-dir", default=str(Path("results")))
    parser.add_argument("--report-out", default=str(Path("reports") / "aef_strong_summary.md"))
    parser.add_argument("--baseline-results", default=str(Path("results") / "counterfactual_2x2.csv"))
    return parser.parse_args()


def build_summary_markdown(
    strong_df: pd.DataFrame,
    control_df: pd.DataFrame,
    series_df: pd.DataFrame,
    baseline_results: pd.DataFrame,
) -> str:
    strong_cmp = compare_with_standard(strong_df, baseline_results)
    control_cmp = compare_with_standard(control_df, baseline_results)
    series_cmp = compare_with_standard(series_df, baseline_results)
    merged = strong_cmp.merge(
        control_cmp,
        on=["dataset_name", "horizon", "eval_view_name"],
        suffixes=("_strong", "_control"),
    ).merge(
        series_cmp,
        on=["dataset_name", "horizon", "eval_view_name"],
        suffixes=("", "_series"),
    )

    lines = [
        "# AEF-Strong Summary",
        "",
        "本轮把 AEF 扩成三路表格 probe：`AEF-Strong / AEF-ControlBlocked / AEF-SeriesOnly`。",
        "",
        "## 结果概览",
        "",
        f"- AEF-Strong setting 数: {len(strong_df)}",
        f"- AEF-ControlBlocked setting 数: {len(control_df)}",
        f"- AEF-SeriesOnly setting 数: {len(series_df)}",
        "",
        "## Falsification Signals",
        "",
    ]
    if merged.empty:
        lines.append("- 当前没有可对齐的 AEF-Strong 结果。")
        return "\n".join(lines)

    raw_rows = merged[merged["eval_view_name"] == "raw"].sort_values("mae_control", ascending=False)
    for row in raw_rows.head(8).itertuples(index=False):
        lines.append(
            f"- raw {row.dataset_name} / H{row.horizon}: "
            f"Strong={row.mae_strong:.4f}, ControlBlocked={row.mae_control:.4f}, "
            f"SeriesOnly={row.mae:.4f}, best_forecaster={row.best_forecaster_mae_strong:.4f}"
        )

    for dataset_name, horizon in sorted(set(zip(merged["dataset_name"], merged["horizon"]))):
        ds_rows = merged[(merged["dataset_name"] == dataset_name) & (merged["horizon"] == horizon)]
        raw_strong = ds_rows.loc[ds_rows["eval_view_name"] == "raw", "mae_strong"]
        int_strong = ds_rows.loc[ds_rows["eval_view_name"] == "intervened", "mae_strong"]
        raw_control = ds_rows.loc[ds_rows["eval_view_name"] == "raw", "mae_control"]
        int_control = ds_rows.loc[ds_rows["eval_view_name"] == "intervened", "mae_control"]
        raw_series = ds_rows.loc[ds_rows["eval_view_name"] == "raw", "mae"]
        int_series = ds_rows.loc[ds_rows["eval_view_name"] == "intervened", "mae"]
        if raw_strong.empty or int_strong.empty or raw_control.empty or int_control.empty or raw_series.empty or int_series.empty:
            continue
        strong_arg = float((int_strong.iloc[0] - raw_strong.iloc[0]) / max(raw_strong.iloc[0], 1e-8))
        control_arg = float((int_control.iloc[0] - raw_control.iloc[0]) / max(raw_control.iloc[0], 1e-8))
        series_arg = float((int_series.iloc[0] - raw_series.iloc[0]) / max(raw_series.iloc[0], 1e-8))
        lines.append(
            f"- {dataset_name} / H{horizon}: "
            f"Strong_ARG={strong_arg:.4f}, ControlBlocked_ARG={control_arg:.4f}, SeriesOnly_ARG={series_arg:.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    runtime_cfg = dict(config["defaults"].get("runtime", {}))
    model_cfg = dict(config["defaults"].get("model", {}))
    feature_cfg = dict(config["defaults"].get("feature", {}))
    lookback = int(config["defaults"].get("lookback", 96))

    results_dir = ROOT_DIR / Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    baseline_results = pd.read_csv(ROOT_DIR / Path(args.baseline_results)) if (ROOT_DIR / Path(args.baseline_results)).exists() else pd.DataFrame()

    bundle_cache: dict[str, Any] = {}
    view_cache: dict[tuple[str, int], pd.DataFrame] = {}
    feature_cache: dict[tuple[str, int, str], tuple[pd.DataFrame, pd.DataFrame]] = {}

    variant_rows: dict[str, list[dict[str, Any]]] = {
        "AEF-Strong": [],
        "AEF-ControlBlocked": [],
        "AEF-SeriesOnly": [],
    }
    variant_errors: dict[str, list[pd.DataFrame]] = {
        "AEF-Strong": [],
        "AEF-ControlBlocked": [],
        "AEF-SeriesOnly": [],
    }

    for dataset_name in config["defaults"]["datasets"]:
        log_progress(f"load dataset={dataset_name}")
        if dataset_name not in bundle_cache:
            bundle_cache[dataset_name] = load_dataset_bundle(dataset_name, registry_path=ROOT_DIR / Path(args.registry))

        for horizon in config["defaults"]["horizons"]:
            log_progress(f"prepare dataset={dataset_name} horizon={horizon}")
            view_key = (dataset_name, int(horizon))
            if view_key not in view_cache:
                view_cache[view_key] = load_view_frame(ROOT_DIR / Path(args.views_dir), dataset_name, lookback=lookback, horizon=int(horizon))
            view_df = view_cache[view_key]
            bundle = bundle_cache[dataset_name]

            train_rows = select_view_rows(
                view_df,
                split_name="train",
                view_name=str(config["defaults"]["train_view_name"]),
                max_rows=runtime_cfg.get("max_train_windows"),
            )
            _, val_rows = resolve_validation_rows(
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

            variant_models: dict[str, Any] = {}
            variant_encoders: dict[str, TabularFeatureEncoder] = {}
            variant_meta: dict[str, tuple[float, int]] = {}
            for idx, model_name in enumerate(["AEF-Strong", "AEF-ControlBlocked", "AEF-SeriesOnly"], start=1):
                log_progress(f"fit dataset={dataset_name} horizon={horizon} model={model_name}")
                model, encoder, best_val_mae, epochs_ran = fit_aef_variant(
                    model_name=model_name,
                    train_features_df=train_features_df,
                    val_features_df=val_features_df,
                    train_targets=train_targets,
                    val_targets=val_targets,
                    train_rows=train_rows,
                    val_rows=val_rows,
                    runtime_cfg=runtime_cfg,
                    model_cfg=model_cfg,
                    seed=100 * idx + int(config["defaults"]["seeds"][0]) + int(horizon),
                    horizon=int(horizon),
                    dataset_name=dataset_name,
                    log_fn=log_progress,
                )
                variant_models[model_name] = model
                variant_encoders[model_name] = encoder
                variant_meta[model_name] = (best_val_mae, epochs_ran)

            device = resolve_device(str(runtime_cfg.get("device", "auto")))
            amp_enabled = bool(runtime_cfg.get("amp", True)) and device.type == "cuda"
            pin_memory = bool(runtime_cfg.get("pin_memory", True)) and device.type == "cuda"

            for eval_view_name in config["defaults"]["eval_views"][dataset_name]:
                cache_key = (dataset_name, int(horizon), str(eval_view_name))
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
                    )
                eval_rows, eval_features_df = feature_cache[cache_key]
                eval_targets = build_targets(eval_rows, bundle=bundle)
                if eval_rows.empty:
                    continue

                for idx, model_name in enumerate(["AEF-Strong", "AEF-ControlBlocked", "AEF-SeriesOnly"], start=1):
                    if model_name == "AEF-Strong":
                        variant_eval_df = eval_features_df.copy()
                    elif model_name == "AEF-ControlBlocked":
                        variant_eval_df = blocked_permute_frame(
                            eval_features_df,
                            cols_to_shuffle=artifact_phase_cols(eval_features_df),
                            block_cols=["cat_dataset_name", "cat_horizon", "cat_dominant_phase_target", "cat_severity_bin"],
                            seed=700 + idx + int(horizon) + len(eval_rows),
                        )
                    else:
                        variant_eval_df = eval_features_df.loc[:, series_only_cols(eval_features_df)].copy()

                    eval_x = variant_encoders[model_name].transform(variant_eval_df)
                    eval_loader = build_loader(
                        TabularForecastDataset(eval_x, eval_targets, eval_rows),
                        batch_size=int(runtime_cfg.get("eval_batch_size", 256)),
                        shuffle=False,
                        num_workers=int(runtime_cfg.get("num_workers", 0)),
                        pin_memory=pin_memory,
                    )
                    metrics, errors = evaluate_aef_model(
                        variant_models[model_name].to(device),
                        eval_loader,
                        device=device,
                        amp_enabled=amp_enabled,
                        meta={
                            "dataset_name": dataset_name,
                            "lookback": int(lookback),
                            "horizon": int(horizon),
                            "train_view_name": str(config["defaults"]["train_view_name"]),
                            "eval_view_name": str(eval_view_name),
                            "model_name": model_name,
                        },
                    )
                    best_val_mae, epochs_ran = variant_meta[model_name]
                    variant_rows[model_name].append(
                        {
                            "dataset_name": dataset_name,
                            "lookback": int(lookback),
                            "horizon": int(horizon),
                            "train_view_name": str(config["defaults"]["train_view_name"]),
                            "eval_view_name": str(eval_view_name),
                            "model_name": model_name,
                            "n_train_windows": int(len(train_rows)),
                            "n_eval_windows": int(len(eval_rows)),
                            "best_val_mae": round(float(best_val_mae), 6),
                            "epochs_ran": int(epochs_ran),
                            "mae": round(float(metrics["mae"]), 6),
                            "mse": round(float(metrics["mse"]), 6),
                            "smape": round(float(metrics["smape"]), 6),
                        }
                    )
                    variant_errors[model_name].append(errors)
                    log_progress(
                        f"done dataset={dataset_name} H{horizon} model={model_name} "
                        f"view={eval_view_name} mae={float(metrics['mae']):.6f}"
                    )

    strong_df = pd.DataFrame(variant_rows["AEF-Strong"])
    control_df = pd.DataFrame(variant_rows["AEF-ControlBlocked"])
    series_df = pd.DataFrame(variant_rows["AEF-SeriesOnly"])
    strong_df.to_csv(results_dir / "aef_strong_results.csv", index=False)
    control_df.to_csv(results_dir / "aef_control_blocked_results.csv", index=False)
    series_df.to_csv(results_dir / "aef_series_only_results.csv", index=False)
    (pd.concat(variant_errors["AEF-Strong"], ignore_index=True) if variant_errors["AEF-Strong"] else pd.DataFrame()).to_csv(
        results_dir / "aef_strong_window_errors.csv",
        index=False,
    )
    (pd.concat(variant_errors["AEF-ControlBlocked"], ignore_index=True) if variant_errors["AEF-ControlBlocked"] else pd.DataFrame()).to_csv(
        results_dir / "aef_control_blocked_window_errors.csv",
        index=False,
    )
    (pd.concat(variant_errors["AEF-SeriesOnly"], ignore_index=True) if variant_errors["AEF-SeriesOnly"] else pd.DataFrame()).to_csv(
        results_dir / "aef_series_only_window_errors.csv",
        index=False,
    )

    write_markdown(ROOT_DIR / Path(args.report_out), build_summary_markdown(strong_df, control_df, series_df, baseline_results))
    log_progress(
        f"finished strong_rows={len(strong_df)} control_rows={len(control_df)} series_rows={len(series_df)}"
    )


if __name__ == "__main__":
    main()
