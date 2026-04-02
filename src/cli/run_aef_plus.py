from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from data import ROOT_DIR, write_markdown
from experiments.aef_shared import compare_with_standard
from experiments.aif_shared import compute_aif_arg_table, compute_aif_wgr_table
from utils.dataset_hparam_presets import resolve_aef_plus_dataset_config
from utils.experiment_profiles import canonicalize_dataset_name
from utils.forecasting_utils import (
    apply_intervention_recipe,
    build_grad_scaler,
    load_dataset_bundle,
    load_events_lookup,
    load_view_frame,
    resolve_device,
    resolve_event_variable_indices,
    resolve_validation_rows,
    select_view_rows,
    set_random_seed,
)


def _load_aefplus_module() -> Any:
    module_path = ROOT_DIR / "baseline" / "AEFPlus" / "AEFPlus.py"
    spec = importlib.util.spec_from_file_location("baseline_aefplus_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load AEFPlus module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_AEFPLUS_MODULE = _load_aefplus_module()
AEFPlus = _AEFPLUS_MODULE.AEFPlus
AEFPlusConfig = _AEFPLUS_MODULE.AEFPlusConfig


def log_progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [012/AEF+] {message}", flush=True)


NUMERIC_METADATA_DIM = 8


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        shadow_state = self.shadow.state_dict()
        for name, value in shadow_state.items():
            source = model_state[name].detach()
            if not torch.is_floating_point(source):
                value.copy_(source)
                continue
            value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)


def parse_optimizer_betas(runtime_cfg: dict[str, Any]) -> tuple[float, float]:
    raw_betas = runtime_cfg.get("betas", (0.9, 0.999))
    if isinstance(raw_betas, (list, tuple)) and len(raw_betas) == 2:
        return float(raw_betas[0]), float(raw_betas[1])
    return 0.9, 0.999


def router_balance_loss(router_weights: torch.Tensor) -> torch.Tensor:
    importance = router_weights.mean(dim=0)
    target = torch.full_like(importance, 1.0 / float(max(importance.numel(), 1)))
    return F.mse_loss(importance, target)


class AEFPlusWindowDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: pd.DataFrame,
        dataset_bundle: Any,
        events_lookup: dict[str, dict[str, Any]],
        artifact_map: dict[str, int],
        phase_map: dict[str, int],
        severity_map: dict[str, int],
        nvar_map: dict[str, int],
        horizon_id: int,
        near_zero_tolerance: float,
        flat_tolerance: float,
        apply_intervention: bool = False,
    ) -> None:
        self.rows = rows.sort_values("target_start").reset_index(drop=True).copy().to_dict(orient="records")
        self.dataset_bundle = dataset_bundle
        self.events_lookup = events_lookup
        self.artifact_map = artifact_map
        self.phase_map = phase_map
        self.severity_map = severity_map
        self.nvar_map = nvar_map
        self.horizon_id = int(horizon_id)
        self.near_zero_tolerance = float(near_zero_tolerance)
        self.flat_tolerance = float(flat_tolerance)
        self.apply_intervention = bool(apply_intervention)
        self.boundary_steps = max(4, min(int(getattr(dataset_bundle, "cycle_len", 24)), 96, 96))

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _safe_text(value: Any, fallback: str = "NA") -> str:
        text = str(value).strip()
        if not text or text == "nan":
            return fallback
        return text

    @staticmethod
    def _safe_float(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            result = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(result):
            return 0.0
        return result

    @staticmethod
    def _parse_artifact_ids(text: Any) -> list[str]:
        value = str(text).strip()
        if not value or value == "nan":
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    def _artifact_stats(self, row: dict[str, Any]) -> tuple[float, float]:
        artifact_ids = self._parse_artifact_ids(row.get("artifact_ids_input", ""))
        if not artifact_ids:
            return 0.0, 0.0
        unique_vars: set[int] = set()
        confidences: list[float] = []
        for artifact_id in artifact_ids:
            event = self.events_lookup.get(artifact_id, {})
            confidences.append(self._safe_float(event.get("confidence", row.get("max_event_weight_input", 0.0))))
            unique_vars.update(
                resolve_event_variable_indices(
                    artifact_id=artifact_id,
                    events_lookup=self.events_lookup,
                    column_index=self.dataset_bundle.column_index,
                )
            )
        flagged_channel_ratio = float(len(unique_vars)) / float(max(1, self.dataset_bundle.n_vars))
        confidence_score = float(np.mean(confidences)) if confidences else 0.0
        return flagged_channel_ratio, confidence_score

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        input_start = int(row["input_start"])
        input_end = int(row["input_end"])
        target_start = int(row["target_start"])
        target_end = int(row["target_end"])

        x = self.dataset_bundle.scaled_values[input_start : input_end + 1].copy()
        y = self.dataset_bundle.scaled_values[target_start : target_end + 1].copy()
        raw_window = self.dataset_bundle.raw_values[input_start : input_end + 1].copy()
        if self.apply_intervention:
            x = apply_intervention_recipe(
                input_window=x,
                global_input_start=input_start,
                recipe_text=str(row.get("intervention_recipe", "")),
                events_lookup=self.events_lookup,
                column_index=self.dataset_bundle.column_index,
            )

        artifact_name = self._safe_text(row.get("artifact_group_major", "NA"))
        phase_name = self._safe_text(row.get("dominant_phase_input", row.get("dominant_phase_target", "NA")))
        severity_name = self._safe_text(row.get("severity_bin", "none"))
        nvar_name = self._safe_text(row.get("n_variables_bin", "NA"))
        flagged_channel_ratio, confidence_score = self._artifact_stats(row)
        zero_ratio = float((np.abs(raw_window) <= self.near_zero_tolerance).mean())
        diffs = np.diff(raw_window, axis=0) if len(raw_window) > 1 else np.zeros((0, raw_window.shape[1]), dtype=np.float32)
        flat_ratio = float((np.abs(diffs) <= self.flat_tolerance).mean()) if diffs.size else 0.0
        tail = raw_window[-min(len(raw_window), max(4, len(raw_window) // 4)) :]
        tail_diffs = np.diff(tail, axis=0) if len(tail) > 1 else np.zeros((0, tail.shape[1]), dtype=np.float32)
        last_k_delta_mean = float(np.abs(tail_diffs).mean()) if tail_diffs.size else 0.0
        distance_to_transition = max(
            0.0,
            1.0 - min(1.0, self._safe_float(row.get("phase_share_input_transition", 0.0)) * 4.0),
        )
        overlap_on_input = self._safe_float(row.get("repairable_input_overlap", 0.0)) + self._safe_float(
            row.get("unrecoverable_input_overlap", 0.0)
        )
        overlap_on_target = max(
            self._safe_float(row.get("target_contam_score", 0.0)),
            self._safe_float(row.get("max_event_weight_target", 0.0)),
        )
        severity_score = max(
            self._safe_float(row.get("input_contam_score", 0.0)),
            self._safe_float(row.get("max_event_weight_input", 0.0)),
            self._safe_float(row.get("max_event_weight_target", 0.0)),
        )
        metadata_num = np.asarray(
            [
                severity_score,
                confidence_score,
                distance_to_transition,
                flagged_channel_ratio,
                zero_ratio,
                flat_ratio,
                last_k_delta_mean,
                overlap_on_input + overlap_on_target,
            ],
            dtype=np.float32,
        )

        return {
            "x_raw": torch.from_numpy(x.astype(np.float32, copy=False)),
            "y": torch.from_numpy(y.astype(np.float32, copy=False)),
            "metadata_num": torch.from_numpy(metadata_num),
            "artifact_id": torch.tensor(self.artifact_map.get(artifact_name, 0), dtype=torch.long),
            "phase_id": torch.tensor(self.phase_map.get(phase_name, 0), dtype=torch.long),
            "severity_bin_id": torch.tensor(self.severity_map.get(severity_name, 0), dtype=torch.long),
            "nvar_bin_id": torch.tensor(self.nvar_map.get(nvar_name, 0), dtype=torch.long),
            "severity_target": torch.tensor(severity_score, dtype=torch.float32),
            "horizon_id": torch.tensor(self.horizon_id, dtype=torch.long),
            "window_id": str(row.get("window_id", "")),
            "group_key": self._safe_text(row.get("primary_group_key", "NA")),
            "phase_group": phase_name,
            "artifact_group_major": artifact_name,
            "is_flagged": int(row.get("is_flagged", 0)),
            "has_input_intervention": int(row.get("has_input_intervention", 0)),
            "strict_target_clean": int(row.get("strict_target_clean", 0)),
            "subset_name": self._safe_text(row.get("subset_name", "")),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AEF-Plus exploit upper-bound training.")
    parser.add_argument("--config", default=str(Path("configs") / "aef_plus.yaml"))
    parser.add_argument("--views-dir", default=str(Path("statistic_results") / "window_views"))
    parser.add_argument("--registry", default=str(Path("statistic_results") / "dataset_registry.csv"))
    parser.add_argument("--events", default=str(Path("statistic_results") / "final_artifact_events.csv"))
    parser.add_argument("--baseline-results", default=str(Path("results") / "counterfactual_2x2.csv"))
    parser.add_argument("--results-out", default=str(Path("results") / "aef_plus_results.csv"))
    parser.add_argument("--window-errors-out", default=str(Path("results") / "aef_plus_window_errors.csv"))
    parser.add_argument("--arg-out", default=str(Path("results") / "aef_plus_artifact_reliance_gap.csv"))
    parser.add_argument("--wgr-out", default=str(Path("results") / "aef_plus_worst_group_risk.csv"))
    parser.add_argument("--report-out", default=str(Path("reports") / "aef_plus_summary.md"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_lookup(values: list[str]) -> dict[str, int]:
    unique = sorted({value if value else "NA" for value in values} | {"NA"})
    return {name: idx for idx, name in enumerate(unique)}


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * max(float(warmup_ratio), 0.0)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_loader(dataset: Dataset[Any], batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def evaluate_aef_plus(
    model: nn.Module,
    dataset: Dataset[Any],
    runtime_cfg: dict[str, Any],
    setting_meta: dict[str, Any] | None,
) -> tuple[dict[str, float], pd.DataFrame]:
    device = next(model.parameters()).device
    amp_enabled = bool(runtime_cfg.get("amp", True)) and device.type == "cuda"
    pin_memory = bool(runtime_cfg.get("pin_memory", True)) and device.type == "cuda"
    loader = build_loader(
        dataset,
        batch_size=int(runtime_cfg.get("eval_batch_size", 128)),
        shuffle=False,
        num_workers=int(runtime_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    total_smape = 0.0
    total_count = 0
    rows: list[dict[str, Any]] = []

    for batch in loader:
        x_raw = batch["x_raw"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        metadata_num = batch["metadata_num"].to(device, non_blocking=True)
        artifact_id = batch["artifact_id"].to(device, non_blocking=True)
        phase_id = batch["phase_id"].to(device, non_blocking=True)
        severity_bin_id = batch["severity_bin_id"].to(device, non_blocking=True)
        nvar_bin_id = batch["nvar_bin_id"].to(device, non_blocking=True)
        horizon_id = batch["horizon_id"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            pred = model.predict(
                x_raw=x_raw,
                metadata_num=metadata_num,
                artifact_id=artifact_id,
                phase_id=phase_id,
                severity_bin_id=severity_bin_id,
                nvar_bin_id=nvar_bin_id,
                horizon_id=horizon_id,
            )
        mae_vec = (pred - y).abs().mean(dim=(1, 2))
        mse_vec = ((pred - y) ** 2).mean(dim=(1, 2))
        smape_vec = (200.0 * (pred - y).abs() / (pred.abs() + y.abs() + 1e-6)).mean(dim=(1, 2))
        batch_size = int(x_raw.shape[0])
        total_mae += float(mae_vec.sum().item())
        total_mse += float(mse_vec.sum().item())
        total_smape += float(smape_vec.sum().item())
        total_count += batch_size

        for idx in range(batch_size):
            rows.append(
                {
                    **(setting_meta or {}),
                    "window_id": str(batch["window_id"][idx]),
                    "group_key": str(batch["group_key"][idx]),
                    "phase_group": str(batch["phase_group"][idx]),
                    "artifact_group_major": str(batch["artifact_group_major"][idx]),
                    "is_flagged": int(batch["is_flagged"][idx]),
                    "has_input_intervention": int(batch["has_input_intervention"][idx]),
                    "strict_target_clean": int(batch["strict_target_clean"][idx]),
                    "subset_name": str(batch["subset_name"][idx]),
                    "mae": float(mae_vec[idx].cpu().item()),
                    "mse": float(mse_vec[idx].cpu().item()),
                    "smape": float(smape_vec[idx].cpu().item()),
                }
            )

    if total_count == 0:
        return {"mae": float("nan"), "mse": float("nan"), "smape": float("nan")}, pd.DataFrame(rows)
    return {
        "mae": float(total_mae / total_count),
        "mse": float(total_mse / total_count),
        "smape": float(total_smape / total_count),
    }, pd.DataFrame(rows)


def fit_aef_plus(
    model: AEFPlus,
    ema: ModelEMA | None,
    train_dataset: Dataset[Any],
    val_raw_dataset: Dataset[Any],
    val_intervened_dataset: Dataset[Any] | None,
    runtime_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    seed: int,
    log_prefix: str,
) -> tuple[float, float, float, int]:
    set_random_seed(seed)
    device = next(model.parameters()).device
    amp_enabled = bool(runtime_cfg.get("amp", True)) and device.type == "cuda"
    scaler = build_grad_scaler(device, enabled=amp_enabled)
    pin_memory = bool(runtime_cfg.get("pin_memory", True)) and device.type == "cuda"
    train_loader = build_loader(
        train_dataset,
        batch_size=int(runtime_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(runtime_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(runtime_cfg.get("lr", 3e-4)),
        weight_decay=float(runtime_cfg.get("weight_decay", 0.05)),
        betas=parse_optimizer_betas(runtime_cfg),
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=max(1, int(runtime_cfg.get("epochs", 30)) * max(1, len(train_loader))),
        warmup_ratio=float(runtime_cfg.get("warmup_ratio", 0.05)),
    )
    grad_clip = float(runtime_cfg.get("grad_clip", 1.0))
    best_state = copy.deepcopy((ema.shadow if ema is not None else model).state_dict())
    best_val_mae = float("inf")
    best_val_mse = float("inf")
    best_val_gap = float("-inf")
    patience_counter = 0
    epochs_ran = 0
    log_progress(
        f"{log_prefix} fit start train_rows={len(train_dataset)} val_raw_rows={len(val_raw_dataset)} "
        f"val_intervened_rows={0 if val_intervened_dataset is None else len(val_intervened_dataset)}"
    )

    lambda_group = float(loss_cfg.get("lambda_group", 0.2))
    lambda_phase = float(loss_cfg.get("lambda_phase", 0.1))
    lambda_severity = float(loss_cfg.get("lambda_severity", 0.05))
    lambda_router_balance = float(loss_cfg.get("lambda_router_balance", 0.0))

    for epoch in range(1, int(runtime_cfg.get("epochs", 30)) + 1):
        model.train()
        for batch in train_loader:
            x_raw = batch["x_raw"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            metadata_num = batch["metadata_num"].to(device, non_blocking=True)
            artifact_id = batch["artifact_id"].to(device, non_blocking=True)
            phase_id = batch["phase_id"].to(device, non_blocking=True)
            severity_bin_id = batch["severity_bin_id"].to(device, non_blocking=True)
            nvar_bin_id = batch["nvar_bin_id"].to(device, non_blocking=True)
            severity_target = batch["severity_target"].to(device, non_blocking=True)
            horizon_id = batch["horizon_id"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(
                    x_raw=x_raw,
                    metadata_num=metadata_num,
                    artifact_id=artifact_id,
                    phase_id=phase_id,
                    severity_bin_id=severity_bin_id,
                    nvar_bin_id=nvar_bin_id,
                    horizon_id=horizon_id,
                )
                pred_loss = F.mse_loss(outputs["pred"], y)
                group_loss = F.cross_entropy(outputs["group_logits"], artifact_id)
                phase_loss = F.cross_entropy(outputs["phase_logits"], phase_id)
                severity_loss = F.mse_loss(outputs["severity_pred"], severity_target)
                balance_loss = pred_loss.new_tensor(0.0)
                if lambda_router_balance > 0.0:
                    balance_loss = router_balance_loss(outputs["router_weights"])
                loss = (
                    pred_loss
                    + lambda_router_balance * balance_loss
                    + lambda_group * group_loss
                    + lambda_phase * phase_loss
                    + lambda_severity * severity_loss
                )

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)

        eval_model = ema.shadow if ema is not None else model
        val_raw_metrics, _ = evaluate_aef_plus(eval_model, val_raw_dataset, runtime_cfg=runtime_cfg, setting_meta=None)
        val_intervened_mae = float("nan")
        val_gap = float("-inf")
        if val_intervened_dataset is not None and len(val_intervened_dataset) > 0:
            val_intervened_metrics, _ = evaluate_aef_plus(
                eval_model,
                val_intervened_dataset,
                runtime_cfg=runtime_cfg,
                setting_meta=None,
            )
            val_intervened_mae = float(val_intervened_metrics["mae"])
            val_gap = float(val_intervened_mae - float(val_raw_metrics["mae"]))
        epochs_ran = epoch
        current_val_mae = float(val_raw_metrics["mae"])
        current_val_mse = float(val_raw_metrics["mse"])
        log_progress(
            f"{log_prefix} epoch {epoch}/{int(runtime_cfg.get('epochs', 30))} "
            f"val_raw_mse={current_val_mse:.6f} best_val_mse={min(best_val_mse, current_val_mse):.6f} "
            f"val_raw_mae={current_val_mae:.6f} "
            f"val_gap={val_gap if math.isfinite(val_gap) else float('nan'):.6f}"
        )

        better_raw = current_val_mse + 1e-6 < best_val_mse
        tie_better_gap = abs(current_val_mse - best_val_mse) <= 1e-6 and val_gap > best_val_gap + 1e-6
        if better_raw or tie_better_gap:
            best_val_mse = current_val_mse
            best_val_mae = current_val_mae
            best_val_gap = val_gap
            best_state = copy.deepcopy((ema.shadow if ema is not None else model).state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= int(runtime_cfg.get("patience", 6)):
                break

    if ema is not None:
        ema.shadow.load_state_dict(best_state)
        model.load_state_dict(best_state)
    else:
        model.load_state_dict(best_state)
    log_progress(
        f"{log_prefix} fit done epochs_ran={epochs_ran} best_val_mse={best_val_mse:.6f} "
        f"best_val_mae={best_val_mae:.6f} "
        f"best_val_gap={best_val_gap if math.isfinite(best_val_gap) else float('nan'):.6f}"
    )
    return best_val_mse, best_val_mae, best_val_gap, epochs_ran


def build_summary_markdown(
    results_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    arg_df: pd.DataFrame,
    wgr_df: pd.DataFrame,
) -> str:
    merged = compare_with_standard(results_df, baseline_df)
    lines = [
        "# AEF-Plus Summary",
        "",
        "本轮新增了 `AEF-Plus` upper-bound exploit 模型，包含 TimeMixer++ 风格 trunk、boundary encoder、metadata token encoder、cross-gated fusion 和 sparse expert head。",
        "",
        "## Legacy-Raw Board",
        "",
    ]
    if not merged.empty:
        raw_rows = merged[merged["eval_view_name"] == "raw"].copy()
        raw_rows["delta_vs_best_forecaster"] = raw_rows["mae"] - raw_rows["best_forecaster_mae"]
        for row in raw_rows.sort_values("delta_vs_best_forecaster").head(8).itertuples(index=False):
            lines.append(
                f"- {row.dataset_name} / L{int(row.lookback)} / H{row.horizon}: "
                f"AEFPlus={row.mae:.4f}, best_forecaster={row.best_forecaster_mae:.4f}, "
                f"delta={row.delta_vs_best_forecaster:.4f}"
            )
    else:
        lines.append("- 当前没有可与标准 forecaster 对齐的 AEF-Plus 结果。")
    lines.extend(["", "## Robustness Board", ""])
    if not arg_df.empty:
        for row in arg_df.sort_values("ARG_mae", ascending=False).head(8).itertuples(index=False):
            lines.append(f"- ARG {row.dataset_name} / L{int(row.lookback)} / H{row.horizon}: {row.ARG_mae:.4f}")
    if not wgr_df.empty:
        for row in wgr_df.sort_values("WGR_gap", ascending=False).head(6).itertuples(index=False):
            lines.append(
                f"- WGR {row.dataset_name} / L{int(row.lookback)} / H{row.horizon} / {row.eval_view_name}: "
                f"{row.WGR_gap:.4f} (worst_group={row.worst_group})"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_config(ROOT_DIR / Path(args.config))
    defaults = dict(config.get("defaults", {}))
    feature_cfg = dict(defaults.get("feature", {}))

    datasets = [canonicalize_dataset_name(str(name)) for name in defaults.get("datasets", [])]
    horizons = [int(item) for item in defaults.get("horizons", [])]
    seeds = [int(item) for item in defaults.get("seeds", [0])]
    horizon_id_map = {int(value): idx for idx, value in enumerate(horizons)}
    views_dir = ROOT_DIR / Path(args.views_dir)
    registry_path = ROOT_DIR / Path(args.registry)
    events_path = ROOT_DIR / Path(args.events)
    baseline_results_path = ROOT_DIR / Path(args.baseline_results)

    bundle_cache: dict[str, Any] = {}
    events_cache: dict[str, dict[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    error_frames: list[pd.DataFrame] = []
    log_progress(f"start datasets={datasets} horizons={horizons}")

    for dataset_name in datasets:
        dataset_cfg = resolve_aef_plus_dataset_config(defaults, dataset_name)
        runtime_cfg = dict(dataset_cfg["runtime"])
        model_cfg = dict(dataset_cfg["model"])
        loss_cfg = dict(dataset_cfg["loss"])
        lookback = int(dataset_cfg["lookback"])
        collapse_aux_label_vocabs = bool(dataset_cfg["collapse_aux_label_vocabs"])
        bundle_cache[dataset_name] = load_dataset_bundle(dataset_name, registry_path=registry_path)
        events_cache[dataset_name] = load_events_lookup(events_path=events_path, dataset_name=dataset_name)
        for horizon in horizons:
            log_progress(f"prepare dataset={dataset_name} L{lookback} H{horizon}")
            view_df = load_view_frame(views_dir, dataset_name, lookback=lookback, horizon=horizon)
            train_rows = select_view_rows(
                view_df,
                split_name="train",
                view_name="raw",
                max_rows=runtime_cfg.get("max_train_windows"),
            )
            val_view_name, val_rows = resolve_validation_rows(
                view_df,
                train_view_name="raw",
                max_val_rows=runtime_cfg.get("max_val_windows"),
            )
            val_intervened_rows = select_view_rows(
                view_df,
                split_name="val",
                view_name="intervened",
                max_rows=runtime_cfg.get("max_val_windows"),
            )
            if train_rows.empty or val_rows.empty:
                log_progress(f"skip dataset={dataset_name} L{lookback} H{horizon} due to empty train/val rows")
                continue

            artifact_map = (
                {"NA": 0}
                if collapse_aux_label_vocabs
                else build_lookup(view_df["artifact_group_major"].fillna("NA").astype(str).tolist())
            )
            phase_map = (
                {"NA": 0}
                if collapse_aux_label_vocabs
                else build_lookup(
                    view_df.get("dominant_phase_input", view_df.get("dominant_phase_target", pd.Series(dtype=str)))
                    .fillna("NA")
                    .astype(str)
                    .tolist()
                )
            )
            severity_map = (
                {"none": 0}
                if collapse_aux_label_vocabs
                else build_lookup(view_df["severity_bin"].fillna("none").astype(str).tolist())
            )
            nvar_map = build_lookup(view_df["n_variables_bin"].fillna("NA").astype(str).tolist())

            for seed in seeds:
                set_random_seed(seed)
                model = AEFPlus(
                    AEFPlusConfig(
                        seq_len=lookback,
                        pred_len=horizon,
                        enc_in=bundle_cache[dataset_name].n_vars,
                        artifact_vocab_size=len(artifact_map),
                        phase_vocab_size=len(phase_map),
                        severity_vocab_size=len(severity_map),
                        nvar_vocab_size=len(nvar_map),
                        horizon_vocab_size=max(len(horizon_id_map), 1),
                        metadata_num_dim=NUMERIC_METADATA_DIM,
                        d_model=int(model_cfg.get("d_model", 192)),
                        metadata_dim=int(model_cfg.get("metadata_dim", 32)),
                        expert_hidden=int(model_cfg.get("expert_hidden", 256)),
                        head_rank=int(model_cfg.get("head_rank", 24)),
                        patch_len=int(model_cfg.get("patch_len", 8)),
                        patch_stride=int(model_cfg.get("patch_stride", 4)),
                        n_blocks=int(model_cfg.get("n_blocks", 3)),
                        n_heads=int(model_cfg.get("n_heads", 4)),
                        ffn_ratio=int(model_cfg.get("ffn_ratio", 3)),
                        dropout=float(model_cfg.get("dropout", 0.1)),
                        stochastic_depth=float(model_cfg.get("stochastic_depth", 0.05)),
                        num_experts=int(model_cfg.get("num_experts", 4)),
                        max_boundary_steps=max(96, int(model_cfg.get("max_boundary_steps", 96))),
                    )
                )
                device = resolve_device(str(runtime_cfg.get("device", "auto")))
                model.to(device)
                ema = ModelEMA(model, decay=float(runtime_cfg.get("ema_decay", 0.999))) if bool(runtime_cfg.get("use_ema", True)) else None

                train_dataset = AEFPlusWindowDataset(
                    rows=train_rows,
                    dataset_bundle=bundle_cache[dataset_name],
                    events_lookup=events_cache[dataset_name],
                    artifact_map=artifact_map,
                    phase_map=phase_map,
                    severity_map=severity_map,
                    nvar_map=nvar_map,
                    horizon_id=horizon_id_map[horizon],
                    near_zero_tolerance=float(feature_cfg.get("near_zero_tolerance", 1e-8)),
                    flat_tolerance=float(feature_cfg.get("flat_tolerance", 1e-8)),
                    apply_intervention=False,
                )
                val_raw_dataset = AEFPlusWindowDataset(
                    rows=val_rows,
                    dataset_bundle=bundle_cache[dataset_name],
                    events_lookup=events_cache[dataset_name],
                    artifact_map=artifact_map,
                    phase_map=phase_map,
                    severity_map=severity_map,
                    nvar_map=nvar_map,
                    horizon_id=horizon_id_map[horizon],
                    near_zero_tolerance=float(feature_cfg.get("near_zero_tolerance", 1e-8)),
                    flat_tolerance=float(feature_cfg.get("flat_tolerance", 1e-8)),
                    apply_intervention=(val_view_name == "intervened"),
                )
                val_intervened_dataset = (
                    AEFPlusWindowDataset(
                        rows=val_intervened_rows,
                        dataset_bundle=bundle_cache[dataset_name],
                        events_lookup=events_cache[dataset_name],
                        artifact_map=artifact_map,
                        phase_map=phase_map,
                        severity_map=severity_map,
                        nvar_map=nvar_map,
                        horizon_id=horizon_id_map[horizon],
                        near_zero_tolerance=float(feature_cfg.get("near_zero_tolerance", 1e-8)),
                        flat_tolerance=float(feature_cfg.get("flat_tolerance", 1e-8)),
                        apply_intervention=True,
                    )
                    if not val_intervened_rows.empty
                    else None
                )

                best_val_mse, best_val_mae, best_val_gap, epochs_ran = fit_aef_plus(
                    model=model,
                    ema=ema,
                    train_dataset=train_dataset,
                    val_raw_dataset=val_raw_dataset,
                    val_intervened_dataset=val_intervened_dataset,
                    runtime_cfg=runtime_cfg,
                    loss_cfg=loss_cfg,
                    seed=seed,
                    log_prefix=f"{dataset_name}/L{lookback}/H{horizon}/seed{seed}",
                )

                eval_model = ema.shadow if ema is not None else model
                for eval_view_name in defaults["eval_views"][dataset_name]:
                    eval_rows = select_view_rows(
                        view_df,
                        split_name="test",
                        view_name=str(eval_view_name),
                        max_rows=runtime_cfg.get("max_test_windows"),
                    )
                    if eval_rows.empty:
                        continue
                    eval_dataset = AEFPlusWindowDataset(
                        rows=eval_rows,
                        dataset_bundle=bundle_cache[dataset_name],
                        events_lookup=events_cache[dataset_name],
                        artifact_map=artifact_map,
                        phase_map=phase_map,
                        severity_map=severity_map,
                        nvar_map=nvar_map,
                        horizon_id=horizon_id_map[horizon],
                        near_zero_tolerance=float(feature_cfg.get("near_zero_tolerance", 1e-8)),
                        flat_tolerance=float(feature_cfg.get("flat_tolerance", 1e-8)),
                        apply_intervention=(str(eval_view_name) == "intervened"),
                    )
                    metrics, errors = evaluate_aef_plus(
                        model=eval_model,
                        dataset=eval_dataset,
                        runtime_cfg=runtime_cfg,
                        setting_meta={
                            "dataset_name": dataset_name,
                            "backbone": "AEFPlus",
                            "lookback": lookback,
                            "horizon": horizon,
                            "train_view_name": "aef_plus",
                            "eval_view_name": str(eval_view_name),
                            "hyperparam_source_kind": dataset_cfg["hyperparam_source_kind"],
                            "hyperparam_source_url": dataset_cfg["hyperparam_source_url"],
                            "hyperparam_source_note": dataset_cfg["hyperparam_source_note"],
                        },
                    )
                    result_rows.append(
                        {
                            "dataset_name": dataset_name,
                            "backbone": "AEFPlus",
                            "lookback": lookback,
                            "horizon": horizon,
                            "train_view_name": "aef_plus",
                            "eval_view_name": str(eval_view_name),
                            "seed": seed,
                            "hyperparam_source_kind": dataset_cfg["hyperparam_source_kind"],
                            "hyperparam_source_url": dataset_cfg["hyperparam_source_url"],
                            "hyperparam_source_note": dataset_cfg["hyperparam_source_note"],
                            "n_train_windows": int(len(train_rows)),
                            "n_eval_windows": int(len(eval_rows)),
                            "best_val_mae": round(float(best_val_mae), 6),
                            "best_val_mse": round(float(best_val_mse), 6),
                            "best_val_gap": round(float(best_val_gap), 6) if math.isfinite(best_val_gap) else np.nan,
                            "epochs_ran": int(epochs_ran),
                            "mae": round(float(metrics["mae"]), 6),
                            "mse": round(float(metrics["mse"]), 6),
                            "smape": round(float(metrics["smape"]), 6),
                        }
                    )
                    error_frames.append(errors)
                    log_progress(
                        f"done dataset={dataset_name} L{lookback} H{horizon} seed={seed} "
                        f"view={eval_view_name} mae={float(metrics['mae']):.6f}"
                    )

    results_df = pd.DataFrame(result_rows)
    window_errors_df = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()
    arg_df = compute_aif_arg_table(results_df)
    wgr_df = compute_aif_wgr_table(window_errors_df)

    results_out = ROOT_DIR / Path(args.results_out)
    window_errors_out = ROOT_DIR / Path(args.window_errors_out)
    arg_out = ROOT_DIR / Path(args.arg_out)
    wgr_out = ROOT_DIR / Path(args.wgr_out)
    report_out = ROOT_DIR / Path(args.report_out)
    for path in [results_out, window_errors_out, arg_out, wgr_out, report_out]:
        path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_out, index=False)
    window_errors_df.to_csv(window_errors_out, index=False)
    arg_df.to_csv(arg_out, index=False)
    wgr_df.to_csv(wgr_out, index=False)

    baseline_df = pd.read_csv(baseline_results_path) if baseline_results_path.exists() else pd.DataFrame()
    write_markdown(report_out, build_summary_markdown(results_df, baseline_df, arg_df, wgr_df))
    log_progress(f"finished results_out={results_out} rows={len(results_df)}")


if __name__ == "__main__":
    main()
