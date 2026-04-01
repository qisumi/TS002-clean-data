from __future__ import annotations

import copy
import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from data import ROOT_DIR, write_markdown
from experiments.aif_shared import (
    build_group_map,
    build_loader,
    compare_against_baseline,
    compute_aif_arg_table,
    compute_aif_ri_table,
    compute_aif_wgr_table,
    group_tail_mean,
    load_config,
    resolve_clean_view_name,
)
from utils.forecasting_utils import (
    apply_intervention_recipe,
    build_grad_scaler,
    evaluate_forecaster,
    forward_backbone,
    instantiate_backbone,
    load_dataset_bundle,
    load_events_lookup,
    load_view_frame,
    resolve_device,
    resolve_validation_rows,
    select_view_rows,
    set_random_seed,
)
from utils.experiment_profiles import canonicalize_dataset_name, resolve_backbone_experiment


def log_progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [008/AIF] {message}", flush=True)


VIEW_FLAG_COLUMNS = {
    "natural_clean": "is_raw_view",
    "clean_like": "is_conservative_clean_view",
    "balanced": "is_phase_balanced_view",
}


class PairedWindowDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: pd.DataFrame,
        dataset_bundle: Any,
        events_lookup: dict[str, dict[str, Any]],
        clean_view_name: str,
        group_map: dict[str, int],
    ) -> None:
        self.rows = rows.sort_values("target_start").reset_index(drop=True).copy().to_dict(orient="records")
        self.dataset_bundle = dataset_bundle
        self.events_lookup = events_lookup
        self.clean_view_name = clean_view_name
        self.group_map = group_map

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        input_start = int(row["input_start"])
        input_end = int(row["input_end"])
        target_start = int(row["target_start"])
        target_end = int(row["target_end"])
        cycle_len = int(getattr(self.dataset_bundle, "cycle_len", 1))
        cycle_index = int(input_start % max(cycle_len, 1))

        x_raw = self.dataset_bundle.scaled_values[input_start : input_end + 1].copy()
        y = self.dataset_bundle.scaled_values[target_start : target_end + 1].copy()
        x_cf = x_raw.copy()

        clean_flag_col = VIEW_FLAG_COLUMNS.get(self.clean_view_name, "is_conservative_clean_view")
        clean_available = int(row.get(clean_flag_col, 0))
        intervened_available = int(row.get("is_intervened_view", 0))
        cf_available = 0
        cf_source = "raw_only"
        severity = max(
            float(row.get("input_contam_score", 0.0) or 0.0),
            float(row.get("max_event_weight_input", 0.0) or 0.0),
            float(row.get("max_event_weight_target", 0.0) or 0.0),
        )

        if intervened_available == 1:
            x_cf = apply_intervention_recipe(
                input_window=x_raw,
                global_input_start=input_start,
                recipe_text=str(row.get("intervention_recipe", "")),
                events_lookup=self.events_lookup,
                column_index=self.dataset_bundle.column_index,
            )
            cf_available = 1
            cf_source = "intervened"
        elif clean_available == 1:
            cf_available = 1
            cf_source = self.clean_view_name

        return {
            "x_raw": torch.from_numpy(x_raw.astype(np.float32, copy=False)),
            "x_cf": torch.from_numpy(x_cf.astype(np.float32, copy=False)),
            "y": torch.from_numpy(y.astype(np.float32, copy=False)),
            "flagged_mask": torch.tensor(float(row.get("is_flagged", 0)), dtype=torch.float32),
            "cf_available": torch.tensor(float(cf_available), dtype=torch.float32),
            "group_id": torch.tensor(self.group_map.get(str(row.get("primary_group_key", "NA")), 0), dtype=torch.long),
            "cf_source": cf_source,
            "cycle_index": torch.tensor(cycle_index, dtype=torch.long),
            "severity": torch.tensor(severity, dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal AIF-Lite paired-view training.")
    parser.add_argument("--config", default=str(Path("configs") / "aif_lite.yaml"))
    parser.add_argument("--views-dir", default=str(Path("statistic_results") / "window_views"))
    parser.add_argument("--registry", default=str(Path("statistic_results") / "dataset_registry.csv"))
    parser.add_argument("--events", default=str(Path("statistic_results") / "final_artifact_events.csv"))
    parser.add_argument("--baseline-results", default=str(Path("results") / "counterfactual_2x2.csv"))
    parser.add_argument("--results-out", default=str(Path("results") / "aif_lite_results.csv"))
    parser.add_argument("--window-errors-out", default=str(Path("results") / "aif_lite_window_errors.csv"))
    parser.add_argument("--arg-out", default=str(Path("results") / "aif_lite_artifact_reliance_gap.csv"))
    parser.add_argument("--wgr-out", default=str(Path("results") / "aif_lite_worst_group_risk.csv"))
    parser.add_argument("--ri-out", default=str(Path("results") / "aif_lite_ranking_instability.csv"))
    parser.add_argument("--report-out", default=str(Path("reports") / "aif_lite_summary.md"))
    return parser.parse_args()


def soft_worst_group_weights(group_ids: torch.Tensor, losses: torch.Tensor, n_groups: int, eta: float) -> torch.Tensor:
    device = losses.device
    group_mean = torch.zeros(n_groups, device=device)
    valid_mask = torch.zeros(n_groups, device=device)
    for group_idx in range(n_groups):
        mask = group_ids == group_idx
        if mask.any():
            group_mean[group_idx] = losses[mask].mean()
            valid_mask[group_idx] = 1.0
    logits = torch.where(valid_mask > 0, eta * group_mean, torch.full_like(group_mean, -1e6))
    weights = torch.softmax(logits, dim=0)
    return weights[group_ids]


def ramp_value(epoch: int, warmup_epochs: int, ramp_epochs: int, max_value: float) -> float:
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(max_value)
    ratio = min(1.0, float(epoch - warmup_epochs) / float(ramp_epochs))
    return float(max_value) * ratio


def resolve_loss_cfg(loss_cfg: dict[str, Any], backbone_name: str) -> dict[str, Any]:
    resolved = {key: value for key, value in loss_cfg.items() if key != "backbone_overrides"}
    overrides = loss_cfg.get("backbone_overrides", {})
    if isinstance(overrides, dict):
        resolved.update(dict(overrides.get(backbone_name, {})))
    if "alpha_cf_max" not in resolved:
        resolved["alpha_cf_max"] = float(resolved.get("alpha", 1.0))
    if "beta_cons_max" not in resolved:
        resolved["beta_cons_max"] = float(resolved.get("beta", 0.1))
    if "gamma_group_max" not in resolved:
        resolved["gamma_group_max"] = float(resolved.get("gamma", 0.2))
    resolved.setdefault("warmup_epochs", 1)
    resolved.setdefault("ramp_epochs", 0)
    resolved.setdefault("severity_scale", 0.0)
    resolved.setdefault("severity_clip", 3.0)
    resolved.setdefault("group_tail_frac", 0.5)
    resolved.setdefault("detach_cf_for_consistency", True)
    resolved.setdefault("cf_batch_prob", 1.0)
    resolved.setdefault("group_loss_mode", "tail")
    return resolved


def fit_aif_lite(
    backbone_name: str,
    backbone_params: dict[str, Any],
    runtime_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    dataset_bundle: Any,
    events_lookup: dict[str, dict[str, Any]],
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    clean_view_name: str,
    seed: int,
    log_prefix: str = "",
) -> tuple[nn.Module, float, int]:
    set_random_seed(seed)
    current_loss_cfg = resolve_loss_cfg(loss_cfg, backbone_name)
    device = resolve_device(str(runtime_cfg.get("device", "auto")))
    pin_memory = bool(runtime_cfg.get("pin_memory", True)) and device.type == "cuda"
    amp_enabled = bool(runtime_cfg.get("amp", True)) and device.type == "cuda"
    scaler = build_grad_scaler(device, enabled=amp_enabled)

    seq_len = int(train_rows["lookback"].iloc[0])
    pred_len = int(train_rows["horizon"].iloc[0])
    model = instantiate_backbone(
        backbone_name=backbone_name,
        seq_len=seq_len,
        pred_len=pred_len,
        n_vars=dataset_bundle.n_vars,
        params=backbone_params,
        dataset_name=dataset_bundle.dataset_name,
    )
    model.to(device)

    group_map = build_group_map(train_rows)
    train_loader = build_loader(
        PairedWindowDataset(train_rows, dataset_bundle, events_lookup, clean_view_name=clean_view_name, group_map=group_map),
        batch_size=int(runtime_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(runtime_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(runtime_cfg.get("lr", 1e-3)),
        weight_decay=float(runtime_cfg.get("weight_decay", 0.0)),
    )

    best_state: dict[str, Any] | None = None
    best_val_mae = float("inf")
    patience_counter = 0
    epochs_ran = 0
    prefix = f"{log_prefix} " if log_prefix else ""
    log_progress(
        f"{prefix}fit start backbone={backbone_name} device={device.type} "
        f"train_rows={len(train_rows)} val_rows={len(val_rows)} "
        f"warmup={int(current_loss_cfg.get('warmup_epochs', 1))} "
        f"ramp={int(current_loss_cfg.get('ramp_epochs', 0))}"
    )

    for epoch in range(1, int(runtime_cfg.get("epochs", 6)) + 1):
        model.train()
        alpha_t = ramp_value(
            epoch,
            warmup_epochs=int(current_loss_cfg.get("warmup_epochs", 1)),
            ramp_epochs=int(current_loss_cfg.get("ramp_epochs", 0)),
            max_value=float(current_loss_cfg.get("alpha_cf_max", 1.0)),
        )
        beta_t = ramp_value(
            epoch,
            warmup_epochs=int(current_loss_cfg.get("warmup_epochs", 1)),
            ramp_epochs=int(current_loss_cfg.get("ramp_epochs", 0)),
            max_value=float(current_loss_cfg.get("beta_cons_max", 0.1)),
        )
        gamma_t = ramp_value(
            epoch,
            warmup_epochs=int(current_loss_cfg.get("warmup_epochs", 1)),
            ramp_epochs=int(current_loss_cfg.get("ramp_epochs", 0)),
            max_value=float(current_loss_cfg.get("gamma_group_max", 0.2)),
        )
        for batch in train_loader:
            x_raw = batch["x_raw"].to(device, non_blocking=True)
            x_cf = batch["x_cf"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            flagged_mask = batch["flagged_mask"].to(device, non_blocking=True)
            cf_available = batch["cf_available"].to(device, non_blocking=True)
            group_ids = batch["group_id"].to(device, non_blocking=True)
            cycle_index = batch["cycle_index"].to(device, non_blocking=True)
            severity = batch["severity"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                pred_raw = forward_backbone(model, x_raw, cycle_index=cycle_index)
                raw_loss_vec = F.l1_loss(pred_raw, y, reduction="none").mean(dim=(1, 2))
                raw_loss = raw_loss_vec.mean()
                loss = raw_loss

                random_mask = torch.rand_like(cf_available) <= float(current_loss_cfg.get("cf_batch_prob", 1.0))
                active_cf = (cf_available > 0.5) & (flagged_mask > 0.5) & random_mask
                severity_weight = 1.0 + float(current_loss_cfg.get("severity_scale", 0.0)) * severity
                severity_weight = torch.clamp(
                    severity_weight,
                    min=1.0,
                    max=float(current_loss_cfg.get("severity_clip", 3.0)),
                )

                if bool(active_cf.any()) and (alpha_t > 0 or beta_t > 0):
                    pred_cf = forward_backbone(model, x_cf, cycle_index=cycle_index)
                    cf_loss_vec = F.l1_loss(pred_cf, y, reduction="none").mean(dim=(1, 2))
                    if alpha_t > 0:
                        cf_loss = (cf_loss_vec[active_cf] * severity_weight[active_cf]).mean()
                        loss = loss + alpha_t * cf_loss
                    if beta_t > 0:
                        teacher = pred_cf.detach() if bool(current_loss_cfg.get("detach_cf_for_consistency", True)) else pred_cf
                        cons_loss_vec = (pred_raw - teacher).abs().mean(dim=(1, 2))
                        cons_loss = (cons_loss_vec[active_cf] * severity_weight[active_cf]).mean()
                        loss = loss + beta_t * cons_loss

                if gamma_t > 0:
                    if str(current_loss_cfg.get("group_loss_mode", "tail")) == "softmax":
                        group_weights = soft_worst_group_weights(
                            group_ids=group_ids,
                            losses=raw_loss_vec.detach(),
                            n_groups=max(len(group_map), 1),
                            eta=float(current_loss_cfg.get("eta", 2.0)),
                        )
                        group_loss = (group_weights * raw_loss_vec).mean()
                    else:
                        group_loss = group_tail_mean(
                            per_sample_loss=raw_loss_vec,
                            group_ids=group_ids,
                            tail_frac=float(current_loss_cfg.get("group_tail_frac", 0.5)),
                        )
                    loss = loss + gamma_t * group_loss

            scaler.scale(loss).backward()
            if float(runtime_cfg.get("grad_clip", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(runtime_cfg.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()

        val_metrics, _ = evaluate_forecaster(
            model=model,
            dataset_bundle=dataset_bundle,
            events_lookup=events_lookup,
            eval_rows=val_rows,
            runtime_cfg=runtime_cfg,
            apply_intervention=False,
            setting_meta=None,
        )
        epochs_ran = epoch
        current_val = float(val_metrics["mae"])
        log_progress(
            f"{prefix}epoch {epoch}/{int(runtime_cfg.get('epochs', 6))} "
            f"val_mae={current_val:.6f} best_val_mae={min(best_val_mae, current_val):.6f} "
            f"alpha={alpha_t:.3f} beta={beta_t:.3f} gamma={gamma_t:.3f}"
        )
        if current_val + 1e-6 < best_val_mae:
            best_val_mae = current_val
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= int(runtime_cfg.get("patience", 2)):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    log_progress(f"{prefix}fit done epochs_ran={epochs_ran} best_val_mae={best_val_mae:.6f}")
    return model, float(best_val_mae), epochs_ran

def build_summary_markdown(
    aif_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    arg_df: pd.DataFrame,
    wgr_df: pd.DataFrame,
    ri_df: pd.DataFrame,
) -> str:
    merged = compare_against_baseline(aif_df, baseline_df)
    lines = [
        "# AIF-Lite Summary",
        "",
        "本轮把 AIF-Lite 落成了双视图配对的训练包装器，主干已支持 `DLinear / PatchTST / TQNet / iTransformer / ModernTCN / TimeMixer / TimeMixerPP`。",
        "",
        "## 结果概览",
        "",
        f"- completed setting 数: {len(aif_df)}",
        "",
        "## Baseline Comparison",
        "",
    ]
    if not merged.empty:
        merged["delta_mae_vs_baseline"] = merged["mae"] - merged["baseline_mae"]
        for row in merged.sort_values("delta_mae_vs_baseline").head(8).itertuples(index=False):
            lines.append(
                f"- {row.dataset_name} / {row.backbone} / L{int(row.lookback)} / H{row.horizon} / {row.eval_view_name}: "
                f"AIF={row.mae:.4f}, ERM={row.baseline_mae:.4f}, delta={row.delta_mae_vs_baseline:.4f}"
            )

        for (dataset_name, backbone, lookback, horizon), group in merged.groupby(
            ["dataset_name", "backbone", "lookback", "horizon"],
            dropna=False,
        ):
            raw_aif = group.loc[group["eval_view_name"] == "raw", "mae"]
            int_aif = group.loc[group["eval_view_name"] == "intervened", "mae"]
            raw_base = group.loc[group["eval_view_name"] == "raw", "baseline_mae"]
            int_base = group.loc[group["eval_view_name"] == "intervened", "baseline_mae"]
            if raw_aif.empty or int_aif.empty or raw_base.empty or int_base.empty:
                continue
            aif_arg = float((int_aif.iloc[0] - raw_aif.iloc[0]) / max(raw_aif.iloc[0], 1e-8))
            base_arg = float((int_base.iloc[0] - raw_base.iloc[0]) / max(raw_base.iloc[0], 1e-8))
            lines.append(
                f"- ARG {dataset_name} / {backbone} / L{int(lookback)} / H{horizon}: "
                f"AIF={aif_arg:.4f}, ERM={base_arg:.4f}"
            )
    else:
        lines.append("- 当前没有可与 ERM baseline 对齐的 AIF-Lite 结果。")
    lines.extend(["", "## Robustness Outputs", ""])
    if not arg_df.empty:
        for row in arg_df.sort_values("ARG_mae", ascending=False).head(6).itertuples(index=False):
            lines.append(f"- ARG {row.dataset_name} / {row.backbone} / L{int(row.lookback)} / H{row.horizon}: {row.ARG_mae:.4f}")
    if not wgr_df.empty:
        for row in wgr_df.sort_values("WGR_gap", ascending=False).head(6).itertuples(index=False):
            lines.append(
                f"- WGR {row.dataset_name} / {row.backbone} / L{int(row.lookback)} / H{row.horizon} / {row.eval_view_name}: "
                f"{row.WGR_gap:.4f} (worst_group={row.worst_group})"
            )
    if not ri_df.empty:
        for row in ri_df.sort_values("mean_rank_shift", ascending=False).head(6).itertuples(index=False):
            lines.append(
                f"- RI {row.dataset_name} / L{int(row.lookback)} / H{row.horizon} / {row.compare_to}: "
                f"top1_flip={row.top1_flip}, mean_rank_shift={row.mean_rank_shift:.4f}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    runtime_defaults = dict(config["defaults"].get("runtime", {}))
    if config["defaults"].get("lookback") is not None:
        runtime_defaults["lookback_override"] = int(config["defaults"]["lookback"])
    loss_cfg = dict(config["defaults"].get("loss", {}))

    views_dir = ROOT_DIR / Path(args.views_dir)
    registry_path = ROOT_DIR / Path(args.registry)
    events_path = ROOT_DIR / Path(args.events)
    baseline_results_path = ROOT_DIR / Path(args.baseline_results)

    bundle_cache: dict[str, Any] = {}
    events_cache: dict[str, dict[str, Any]] = {}
    view_cache: dict[tuple[str, int, int], pd.DataFrame] = {}
    backbones = [dict(item) for item in config["defaults"]["backbones"]]

    result_rows: list[dict[str, Any]] = []
    error_frames: list[pd.DataFrame] = []
    log_progress(
        f"start datasets={config['defaults']['datasets']} horizons={config['defaults']['horizons']} "
        f"backbones={[item['name'] for item in config['defaults']['backbones']]}"
    )

    for dataset_name_raw in config["defaults"]["datasets"]:
        dataset_name = canonicalize_dataset_name(str(dataset_name_raw))
        log_progress(f"load dataset={dataset_name}")
        bundle_cache[dataset_name] = load_dataset_bundle(dataset_name, registry_path=registry_path)
        events_cache[dataset_name] = load_events_lookup(events_path=events_path, dataset_name=dataset_name)
        clean_view_name = resolve_clean_view_name(config, dataset_name)

        for horizon in config["defaults"]["horizons"]:
            for backbone_cfg in backbones:
                backbone_name = str(backbone_cfg["name"])
                resolved = resolve_backbone_experiment(
                    backbone_cfg=dict(backbone_cfg),
                    dataset_name=dataset_name,
                    horizon=int(horizon),
                    runtime_defaults=runtime_defaults,
                )
                view_key = (dataset_name, int(resolved.lookback), int(horizon))
                if view_key not in view_cache:
                    log_progress(f"prepare dataset={dataset_name} backbone={backbone_name} L{resolved.lookback} H{horizon}")
                    view_cache[view_key] = load_view_frame(
                        views_dir,
                        dataset_name,
                        lookback=int(resolved.lookback),
                        horizon=int(horizon),
                    )
                view_df = view_cache[view_key]

                train_rows = select_view_rows(
                    view_df,
                    split_name="train",
                    view_name="raw",
                    max_rows=resolved.runtime.get("max_train_windows"),
                )
                _, val_rows = resolve_validation_rows(
                    view_df,
                    train_view_name="raw",
                    max_val_rows=resolved.runtime.get("max_val_windows"),
                )
                if train_rows.empty or val_rows.empty:
                    log_progress(
                        f"skip dataset={dataset_name} backbone={backbone_name} "
                        f"L{resolved.lookback} H{horizon} due to empty train/val rows"
                    )
                    continue

                log_progress(f"fit dataset={dataset_name} backbone={backbone_name} L{resolved.lookback} H{horizon}")
                model, best_val_mae, epochs_ran = fit_aif_lite(
                    backbone_name=backbone_name,
                    backbone_params=dict(resolved.model_params),
                    runtime_cfg=dict(resolved.runtime),
                    loss_cfg=loss_cfg,
                    dataset_bundle=bundle_cache[dataset_name],
                    events_lookup=events_cache[dataset_name],
                    train_rows=train_rows,
                    val_rows=val_rows,
                    clean_view_name=clean_view_name,
                    seed=int(config["defaults"]["seeds"][0]),
                    log_prefix=f"{dataset_name}/L{resolved.lookback}/H{horizon}/{backbone_name}",
                )

                for eval_view_name in ["raw", clean_view_name, "intervened"]:
                    log_progress(
                        f"evaluate dataset={dataset_name} backbone={backbone_name} "
                        f"L{resolved.lookback} H{horizon} view={eval_view_name}"
                    )
                    eval_rows = select_view_rows(
                        view_df,
                        split_name="test",
                        view_name=str(eval_view_name),
                        max_rows=resolved.runtime.get("max_test_windows"),
                    )
                    if eval_rows.empty:
                        continue

                    metrics, errors = evaluate_forecaster(
                        model=model,
                        dataset_bundle=bundle_cache[dataset_name],
                        events_lookup=events_cache[dataset_name],
                        eval_rows=eval_rows,
                        runtime_cfg=dict(resolved.runtime),
                        apply_intervention=(eval_view_name == "intervened"),
                        setting_meta={
                            "dataset_name": dataset_name,
                            "backbone": backbone_name,
                            "lookback": int(resolved.lookback),
                            "horizon": int(horizon),
                            "train_view_name": "aif_lite",
                            "eval_view_name": str(eval_view_name),
                            "hyperparam_source_kind": str(resolved.source_kind),
                            "hyperparam_source_url": str(resolved.source_url),
                            "hyperparam_source_note": str(resolved.source_note),
                        },
                    )
                    result_rows.append(
                        {
                            "dataset_name": dataset_name,
                            "backbone": backbone_name,
                            "lookback": int(resolved.lookback),
                            "horizon": int(horizon),
                            "train_view_name": "aif_lite",
                            "eval_view_name": str(eval_view_name),
                            "hyperparam_source_kind": str(resolved.source_kind),
                            "hyperparam_source_url": str(resolved.source_url),
                            "hyperparam_source_note": str(resolved.source_note),
                            "n_train_windows": int(len(train_rows)),
                            "n_eval_windows": int(len(eval_rows)),
                            "best_val_mae": round(float(best_val_mae), 6),
                            "epochs_ran": int(epochs_ran),
                            "mae": round(float(metrics["mae"]), 6),
                            "mse": round(float(metrics["mse"]), 6),
                            "smape": round(float(metrics["smape"]), 6),
                        }
                    )
                    error_frames.append(errors)
                    log_progress(
                        f"done dataset={dataset_name} backbone={backbone_name} "
                        f"L{resolved.lookback} H{horizon} "
                        f"view={eval_view_name} mae={float(metrics['mae']):.6f}"
                    )

    results_df = pd.DataFrame(result_rows)
    window_errors_df = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()
    arg_df = compute_aif_arg_table(results_df)
    wgr_df = compute_aif_wgr_table(window_errors_df)
    ri_df = compute_aif_ri_table(results_df)
    results_out = ROOT_DIR / Path(args.results_out)
    window_errors_out = ROOT_DIR / Path(args.window_errors_out)
    arg_out = ROOT_DIR / Path(args.arg_out)
    wgr_out = ROOT_DIR / Path(args.wgr_out)
    ri_out = ROOT_DIR / Path(args.ri_out)
    report_out = ROOT_DIR / Path(args.report_out)
    results_out.parent.mkdir(parents=True, exist_ok=True)
    window_errors_out.parent.mkdir(parents=True, exist_ok=True)
    arg_out.parent.mkdir(parents=True, exist_ok=True)
    wgr_out.parent.mkdir(parents=True, exist_ok=True)
    ri_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_out, index=False)
    window_errors_df.to_csv(window_errors_out, index=False)
    arg_df.to_csv(arg_out, index=False)
    wgr_df.to_csv(wgr_out, index=False)
    ri_df.to_csv(ri_out, index=False)

    baseline_df = pd.read_csv(baseline_results_path) if baseline_results_path.exists() else pd.DataFrame()
    write_markdown(report_out, build_summary_markdown(results_df, baseline_df, arg_df, wgr_df, ri_df))
    log_progress(f"finished results_out={results_out} rows={len(results_df)}")


if __name__ == "__main__":
    main()
