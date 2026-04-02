from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from data import ROOT_DIR, write_markdown
from data.forecasting import DatasetBundle, load_dataset_bundle, load_events_lookup, load_view_frame
from training import evaluate_forecaster, fit_forecaster
from views import resolve_validation_rows, select_view_rows
from utils.experiment_profiles import resolve_backbone_experiment


def log_progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [011/var-stratified] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ETTh2 variable-stratified auxiliary evaluation.")
    parser.add_argument("--config", default=str(Path("configs") / "counterfactual_eval.yaml"))
    parser.add_argument("--registry", default=str(Path("statistic_results") / "dataset_registry.csv"))
    parser.add_argument("--events", default=str(Path("statistic_results") / "final_artifact_events.csv"))
    parser.add_argument("--views-dir", default=str(Path("statistic_results") / "window_views"))
    parser.add_argument("--support-csv", default=str(Path("results") / "etth2_channel_support.csv"))
    parser.add_argument("--results-out", default=str(Path("results") / "etth2_variable_stratified_eval.csv"))
    parser.add_argument("--report-out", default=str(Path("reports") / "etth2_variable_stratified_eval.md"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_clean_view_name(config: dict[str, Any], dataset_name: str) -> str:
    return str(config["defaults"]["view_alias"][dataset_name]["clean_like"])


def subset_bundle(bundle: DatasetBundle, channel_names: list[str]) -> DatasetBundle:
    idx = [bundle.column_index[name] for name in channel_names if name in bundle.column_index]
    if not idx:
        raise ValueError("channel subset is empty after intersecting with dataset columns")
    picked_names = [bundle.column_names[i] for i in idx]
    return DatasetBundle(
        dataset_name=bundle.dataset_name,
        raw_values=bundle.raw_values[:, idx],
        scaled_values=bundle.scaled_values[:, idx],
        column_names=picked_names,
        column_index={name: i for i, name in enumerate(picked_names)},
        train_mean=bundle.train_mean[idx],
        train_std=bundle.train_std[idx],
        cycle_len=bundle.cycle_len,
    )


def build_summary_markdown(results_df: pd.DataFrame, support_df: pd.DataFrame) -> str:
    lines = [
        "# ETTh2 Variable-Stratified Auxiliary Evaluation",
        "",
        "这份报告把 ETTh2 从“整体 strict support collapse”拆成通道分层后的辅助评测。",
        "",
        "## Channel Status",
        "",
    ]
    if not support_df.empty:
        for horizon, gdf in support_df.groupby("horizon", dropna=False):
            counts = gdf["status"].astype(str).value_counts().to_dict()
            lines.append(f"- H{int(horizon)}: {counts}")
    else:
        lines.append("- support 表为空。")

    lines.extend(["", "## Auxiliary Eval", ""])
    if results_df.empty:
        lines.append("- 当前没有可用的 variable-stratified 结果。")
        return "\n".join(lines)

    for row in results_df.sort_values(["horizon", "channel_subset", "train_view_name", "backbone"]).itertuples(index=False):
        lines.append(
            f"- H{int(row.horizon)} / {row.backbone} / train={row.train_view_name} / subset={row.channel_subset}: "
            f"channels={int(row.n_channels)}, raw={row.mae_raw:.4f}, input_only={row.mae_input_intervened:.4f}, "
            f"ARG={row.ARG_mae:.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_config(ROOT_DIR / Path(args.config))
    runtime_cfg = dict(config["defaults"].get("runtime", {}))
    if config["defaults"].get("lookback") is not None:
        runtime_cfg["lookback_override"] = int(config["defaults"]["lookback"])
    clean_view_name = resolve_clean_view_name(config, "ETTh2")
    backbones = config["defaults"]["backbones"]

    support_df = pd.read_csv(ROOT_DIR / Path(args.support_csv))
    support_df = support_df[support_df["channel"].astype(str).ne("")].copy()
    bundle = load_dataset_bundle("ETTh2", registry_path=ROOT_DIR / Path(args.registry))
    events_lookup = load_events_lookup(ROOT_DIR / Path(args.events), dataset_name="ETTh2")

    rows: list[dict[str, Any]] = []

    for horizon in sorted({int(item) for item in support_df["horizon"].dropna().astype(int).tolist()}):
        horizon_support = support_df[support_df["horizon"] == horizon].copy()
        recoverable_channels = sorted(
            horizon_support.loc[horizon_support["status"].astype(str) == "recoverable", "channel"].astype(str).tolist()
        )
        if not recoverable_channels:
            log_progress(f"skip H{horizon} due to empty recoverable channel set")
            continue

        channel_subsets = {
            "all_channels": list(bundle.column_names),
            "recoverable_only": recoverable_channels,
        }

        for subset_name, channels in channel_subsets.items():
            subset = subset_bundle(bundle, channels)
            for backbone_cfg in backbones:
                backbone_name = str(backbone_cfg["name"])
                resolved = resolve_backbone_experiment(
                    backbone_cfg=dict(backbone_cfg),
                    dataset_name="ETTh2",
                    horizon=horizon,
                    runtime_defaults=runtime_cfg,
                )
                view_df = load_view_frame(
                    ROOT_DIR / Path(args.views_dir),
                    "ETTh2",
                    lookback=int(resolved.lookback),
                    horizon=horizon,
                )
                eval_rows = select_view_rows(
                    view_df,
                    split_name="test",
                    view_name="raw",
                    max_rows=resolved.runtime.get("max_test_windows"),
                )
                if eval_rows.empty:
                    log_progress(f"skip H{horizon} backbone={backbone_name} due to empty raw test rows")
                    continue
                for train_view_name in ["raw", clean_view_name]:
                    train_rows = select_view_rows(
                        view_df,
                        split_name="train",
                        view_name=train_view_name,
                        max_rows=resolved.runtime.get("max_train_windows"),
                    )
                    _, val_rows = resolve_validation_rows(
                        view_df,
                        train_view_name=train_view_name,
                        max_val_rows=resolved.runtime.get("max_val_windows"),
                    )
                    if train_rows.empty or val_rows.empty:
                        log_progress(
                            f"skip H{horizon} subset={subset_name} backbone={backbone_name} train={train_view_name} "
                            f"due to empty train/val"
                        )
                        continue

                    prefix = f"ETTh2/H{horizon}/{subset_name}/{backbone_name}/{train_view_name}"
                    log_progress(f"fit {prefix} channels={channels}")
                    artifacts = fit_forecaster(
                        backbone_name=backbone_name,
                        model_params=dict(resolved.model_params),
                        runtime_cfg=dict(resolved.runtime),
                        dataset_bundle=subset,
                        events_lookup=events_lookup,
                        train_rows=train_rows,
                        val_rows=val_rows,
                        train_view_name=train_view_name,
                        val_view_name=train_view_name,
                        seed=int(config["defaults"]["seeds"][0]),
                        log_prefix=prefix,
                    )
                    raw_metrics, _ = evaluate_forecaster(
                        model=artifacts.model,
                        dataset_bundle=subset,
                        events_lookup=events_lookup,
                        eval_rows=eval_rows,
                        runtime_cfg=dict(resolved.runtime),
                        apply_intervention=False,
                        setting_meta=None,
                    )
                    cf_metrics, _ = evaluate_forecaster(
                        model=artifacts.model,
                        dataset_bundle=subset,
                        events_lookup=events_lookup,
                        eval_rows=eval_rows,
                        runtime_cfg=dict(resolved.runtime),
                        apply_intervention=True,
                        setting_meta=None,
                    )
                    rows.append(
                        {
                            "dataset_name": "ETTh2",
                            "lookback": int(resolved.lookback),
                            "horizon": horizon,
                            "backbone": backbone_name,
                            "train_view_name": train_view_name,
                            "channel_subset": subset_name,
                            "n_channels": int(len(channels)),
                            "channels": ",".join(channels),
                            "hyperparam_source_kind": str(resolved.source_kind),
                            "hyperparam_source_url": str(resolved.source_url),
                            "hyperparam_source_note": str(resolved.source_note),
                            "best_val_mae": round(float(artifacts.best_val_mae), 6),
                            "best_val_mse": round(float(artifacts.best_val_mse), 6),
                            "epochs_ran": int(artifacts.epochs_ran),
                            "fit_seconds": round(float(artifacts.fit_seconds), 3),
                            "mae_raw": round(float(raw_metrics["mae"]), 6),
                            "mae_input_intervened": round(float(cf_metrics["mae"]), 6),
                            "ARG_mae": round(
                                float((float(cf_metrics["mae"]) - float(raw_metrics["mae"])) / max(float(raw_metrics["mae"]), 1e-8)),
                                6,
                            ),
                            "mse_raw": round(float(raw_metrics["mse"]), 6),
                            "mse_input_intervened": round(float(cf_metrics["mse"]), 6),
                            "ARG_mse": round(
                                float((float(cf_metrics["mse"]) - float(raw_metrics["mse"])) / max(float(raw_metrics["mse"]), 1e-8)),
                                6,
                            ),
                        }
                    )
                    log_progress(
                        f"done {prefix} raw_mae={float(raw_metrics['mae']):.6f} "
                        f"input_only_mae={float(cf_metrics['mae']):.6f}"
                    )

    results_df = pd.DataFrame(rows)
    results_path = ROOT_DIR / Path(args.results_out)
    report_path = ROOT_DIR / Path(args.report_out)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    write_markdown(report_path, build_summary_markdown(results_df, support_df))
    log_progress(f"wrote results rows={len(results_df)} -> {results_path}")


if __name__ == "__main__":
    main()
