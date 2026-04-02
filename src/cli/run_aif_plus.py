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
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from data import ROOT_DIR, write_markdown
from experiments.aif_shared import (
    build_group_map,
    build_loader,
    build_lookup,
    compare_against_baseline,
    compute_aif_arg_table,
    compute_aif_ri_table,
    compute_aif_wgr_table,
    group_tail_mean,
    load_config,
    resolve_clean_view_name,
)
from utils.dataset_hparam_presets import resolve_aif_plus_dataset_config
from utils.experiment_profiles import canonicalize_dataset_name
from utils.forecasting_utils import (
    apply_intervention_recipe,
    build_grad_scaler,
    load_dataset_bundle,
    load_events_lookup,
    load_view_frame,
    parse_intervention_recipe,
    resolve_device,
    resolve_event_variable_indices,
    select_view_rows,
    set_random_seed,
)


def _load_aifplus_module() -> Any:
    module_path = ROOT_DIR / "baseline" / "AIFPlus" / "AIFPlus.py"
    spec = importlib.util.spec_from_file_location("baseline_aifplus_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load AIFPlus module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_AIFPLUS_MODULE = _load_aifplus_module()
AIFPlus = _AIFPLUS_MODULE.AIFPlus
AIFPlusConfig = _AIFPLUS_MODULE.AIFPlusConfig


def log_progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [AIF+] {message}", flush=True)


VIEW_FLAG_COLUMNS = {
    "natural_clean": "is_raw_view",
    "clean_like": "is_conservative_clean_view",
    "balanced": "is_phase_balanced_view",
}

METADATA_NUMERIC_COLUMNS = [
    "input_contam_score",
    "target_contam_score",
    "n_events_input",
    "max_event_weight_input",
    "max_event_weight_target",
    "repairable_input_overlap",
    "unrecoverable_input_overlap",
    "phase_share_input_active",
    "phase_share_input_transition",
    "phase_share_input_night",
    "is_flagged",
    "soft_mask_mean",
]


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

    def state_dict(self) -> dict[str, Any]:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.shadow.load_state_dict(state_dict)


def parse_optimizer_betas(runtime_cfg: dict[str, Any]) -> tuple[float, float]:
    raw_betas = runtime_cfg.get("betas", (0.9, 0.999))
    if isinstance(raw_betas, (list, tuple)) and len(raw_betas) == 2:
        return float(raw_betas[0]), float(raw_betas[1])
    return 0.9, 0.999


def router_balance_loss(router_weights: torch.Tensor) -> torch.Tensor:
    importance = router_weights.mean(dim=0)
    target = torch.full_like(importance, 1.0 / float(max(importance.numel(), 1)))
    return F.mse_loss(importance, target)


class AIFPlusWindowDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: pd.DataFrame,
        dataset_bundle: Any,
        events_lookup: dict[str, dict[str, Any]],
        clean_view_name: str,
        group_map: dict[str, int],
        artifact_map: dict[str, int],
        phase_map: dict[str, int],
        dataset_id: int,
        support_id_lookup: dict[str, int],
        horizon_id: int,
        split_name: str,
        synthetic_pretrain: bool = False,
        force_intervened_input: bool = False,
        synthetic_prob: float = 0.0,
        synthetic_seed: int = 0,
    ) -> None:
        self.rows = rows.sort_values("target_start").reset_index(drop=True).copy().to_dict(orient="records")
        self.dataset_bundle = dataset_bundle
        self.events_lookup = events_lookup
        self.clean_view_name = clean_view_name
        self.group_map = group_map
        self.artifact_map = artifact_map
        self.phase_map = phase_map
        self.dataset_id = int(dataset_id)
        self.support_id = int(support_id_lookup.get(split_name, 0))
        self.horizon_id = int(horizon_id)
        self.synthetic_pretrain = bool(synthetic_pretrain)
        self.force_intervened_input = bool(force_intervened_input)
        self.synthetic_prob = float(synthetic_prob)
        self.rng = np.random.default_rng(int(synthetic_seed))

    def __len__(self) -> int:
        return len(self.rows)

    def _soft_mask_from_recipe(
        self,
        row: dict[str, Any],
        x_raw: np.ndarray,
        input_start: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = np.zeros_like(x_raw, dtype=np.float32)
        uncertainty = np.zeros_like(x_raw, dtype=np.float32)
        x_interp = x_raw.copy()
        recipe_text = str(row.get("intervention_recipe", ""))
        if not recipe_text or recipe_text == "nan":
            return x_interp, mask, uncertainty

        x_interp = apply_intervention_recipe(
            input_window=x_raw,
            global_input_start=input_start,
            recipe_text=recipe_text,
            events_lookup=self.events_lookup,
            column_index=self.dataset_bundle.column_index,
        )

        input_end = input_start + x_raw.shape[0] - 1
        for op in parse_intervention_recipe(recipe_text):
            op_name = str(op.get("op", ""))
            if op_name == "drop_window":
                continue
            span_start = max(int(op.get("start", input_start)), input_start)
            span_end = min(int(op.get("end", input_end)), input_end)
            if span_start > span_end:
                continue
            local_start = span_start - input_start
            local_end = span_end - input_start
            artifact_id = str(op.get("artifact_id", ""))
            event = self.events_lookup.get(artifact_id, {})
            indices = resolve_event_variable_indices(
                artifact_id=artifact_id,
                events_lookup=self.events_lookup,
                column_index=self.dataset_bundle.column_index,
            )
            confidence = float(event.get("confidence", row.get("max_event_weight_input", 0.5)) or 0.5)
            severity = float(event.get("severity", row.get("input_contam_score", 0.0)) or 0.0)
            weight = np.clip(0.20 + 0.45 * confidence + 0.35 * severity, 0.0, 0.95)
            uncert = np.clip(1.0 - confidence + 0.25 * severity, 0.0, 1.0)
            mask[local_start : local_end + 1, indices] = np.maximum(mask[local_start : local_end + 1, indices], weight)
            uncertainty[local_start : local_end + 1, indices] = np.maximum(
                uncertainty[local_start : local_end + 1, indices],
                uncert,
            )
        return x_interp, mask, uncertainty

    def _inject_synthetic_artifact(
        self,
        x_clean: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_corrupt = x_clean.copy()
        seq_len, n_vars = x_clean.shape
        mask = np.zeros_like(x_clean, dtype=np.float32)
        uncertainty = np.zeros_like(x_clean, dtype=np.float32)
        span_len = int(self.rng.integers(max(4, seq_len // 12), max(5, seq_len // 4 + 1)))
        start = int(self.rng.integers(0, max(1, seq_len - span_len + 1)))
        end = min(seq_len, start + span_len)
        width = int(self.rng.integers(1, max(2, min(n_vars, max(2, n_vars // 4)) + 1)))
        dims = self.rng.choice(n_vars, size=width, replace=False)
        op_name = str(self.rng.choice(["zero_block", "flat_run", "near_constant", "suspicious_repetition"]))
        if op_name == "zero_block":
            x_corrupt[start:end, dims] = 0.0
        elif op_name == "flat_run":
            source = x_corrupt[start - 1 : start, dims] if start > 0 else x_corrupt[start : start + 1, dims]
            x_corrupt[start:end, dims] = source
        elif op_name == "near_constant":
            base = x_clean[max(0, start - 4) : min(seq_len, end + 4), dims].mean(axis=0, keepdims=True)
            noise = 0.02 * self.rng.standard_normal(size=(end - start, width)).astype(np.float32)
            x_corrupt[start:end, dims] = base + noise
        else:
            if start >= 4:
                src_len = min(end - start, start)
                source = x_clean[start - src_len : start, dims]
                if len(source) == 0:
                    source = x_clean[start : start + 1, dims]
                if len(source) < (end - start):
                    repeat_factor = int(math.ceil((end - start) / max(len(source), 1)))
                    source = np.tile(source, (repeat_factor, 1))
                x_corrupt[start:end, dims] = source[: end - start]
            else:
                x_corrupt[start:end, dims] = np.repeat(x_clean[start : start + 1, dims], end - start, axis=0)
        mask[start:end, dims] = 0.85
        uncertainty[start:end, dims] = 0.75
        x_masked = (1.0 - mask) * x_corrupt + mask * x_clean
        return x_corrupt, x_masked, mask, uncertainty

    def _safe_float(self, value: Any) -> float:
        if value is None:
            return 0.0
        try:
            result = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(result):
            return 0.0
        return result

    def _safe_text(self, value: Any, fallback: str = "NA") -> str:
        text = str(value).strip()
        if not text or text == "nan":
            return fallback
        return text

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        input_start = int(row["input_start"])
        input_end = int(row["input_end"])
        target_start = int(row["target_start"])
        target_end = int(row["target_end"])

        x_base = self.dataset_bundle.scaled_values[input_start : input_end + 1].copy()
        y = self.dataset_bundle.scaled_values[target_start : target_end + 1].copy()
        x_cf, rec_mask, uncertainty = self._soft_mask_from_recipe(row=row, x_raw=x_base, input_start=input_start)
        x_raw = x_base.copy()

        if self.synthetic_pretrain and self.synthetic_prob > 0.0 and float(self.rng.random()) <= self.synthetic_prob:
            x_raw, x_masked, rec_mask, uncertainty = self._inject_synthetic_artifact(x_clean=x_base)
            x_cf = x_base.copy()
        elif self.force_intervened_input:
            x_raw = x_cf.copy()
            x_masked = x_cf.copy()
            rec_mask = np.zeros_like(x_base, dtype=np.float32)
            uncertainty = np.zeros_like(x_base, dtype=np.float32)
        else:
            x_masked = (1.0 - rec_mask) * x_raw + rec_mask * x_cf

        clean_flag_col = VIEW_FLAG_COLUMNS.get(self.clean_view_name, "is_conservative_clean_view")
        pair_available = int(
            int(row.get("is_flagged", 0)) == 1
            and int(row.get("is_intervened_view", 0)) == 1
            and (
                int(row.get(clean_flag_col, 0)) == 1
                or int(row.get("strict_target_clean", 0)) == 1
            )
        )
        artifact_name = self._safe_text(row.get("artifact_group_major", "NA"))
        phase_name = self._safe_text(row.get("dominant_phase_input", row.get("dominant_phase_target", "NA")))
        metadata_num = np.asarray(
            [
                self._safe_float(row.get("input_contam_score", 0.0)),
                self._safe_float(row.get("target_contam_score", 0.0)),
                self._safe_float(row.get("n_events_input", 0.0)),
                self._safe_float(row.get("max_event_weight_input", 0.0)),
                self._safe_float(row.get("max_event_weight_target", 0.0)),
                self._safe_float(row.get("repairable_input_overlap", 0.0)),
                self._safe_float(row.get("unrecoverable_input_overlap", 0.0)),
                self._safe_float(row.get("phase_share_input_active", 0.0)),
                self._safe_float(row.get("phase_share_input_transition", 0.0)),
                self._safe_float(row.get("phase_share_input_night", 0.0)),
                self._safe_float(row.get("is_flagged", 0.0)),
                float(rec_mask.mean()) if rec_mask.size else 0.0,
            ],
            dtype=np.float32,
        )

        return {
            "x_raw": torch.from_numpy(x_raw.astype(np.float32, copy=False)),
            "x_masked": torch.from_numpy(x_masked.astype(np.float32, copy=False)),
            "x_cf": torch.from_numpy(x_cf.astype(np.float32, copy=False)),
            "uncertainty": torch.from_numpy(uncertainty.astype(np.float32, copy=False)),
            "rec_mask": torch.from_numpy(rec_mask.astype(np.float32, copy=False)),
            "rec_target": torch.from_numpy(x_cf.astype(np.float32, copy=False)),
            "y": torch.from_numpy(y.astype(np.float32, copy=False)),
            "metadata_num": torch.from_numpy(metadata_num),
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.long),
            "support_id": torch.tensor(self.support_id, dtype=torch.long),
            "horizon_id": torch.tensor(self.horizon_id, dtype=torch.long),
            "artifact_id": torch.tensor(self.artifact_map.get(artifact_name, 0), dtype=torch.long),
            "phase_id": torch.tensor(self.phase_map.get(phase_name, 0), dtype=torch.long),
            "group_id": torch.tensor(self.group_map.get(self._safe_text(row.get("primary_group_key", "NA")), 0), dtype=torch.long),
            "pair_available": torch.tensor(float(pair_available), dtype=torch.float32),
            "flagged_mask": torch.tensor(float(row.get("is_flagged", 0)), dtype=torch.float32),
            "window_id": str(row.get("window_id", "")),
            "group_key": self._safe_text(row.get("primary_group_key", "NA")),
            "phase_group": phase_name,
            "artifact_group_major": artifact_name,
            "has_input_intervention": int(row.get("has_input_intervention", 0)),
            "strict_target_clean": int(row.get("strict_target_clean", 0)),
            "subset_name": self._safe_text(row.get("subset_name", "")),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIF-Plus clean-view training.")
    parser.add_argument("--config", default=str(Path("configs") / "aif_plus.yaml"))
    parser.add_argument("--views-dir", default=str(Path("statistic_results") / "window_views"))
    parser.add_argument("--registry", default=str(Path("statistic_results") / "dataset_registry.csv"))
    parser.add_argument("--events", default=str(Path("statistic_results") / "final_artifact_events.csv"))
    parser.add_argument("--support-summary", default=str(Path("reports") / "clean_view_support_summary.csv"))
    parser.add_argument("--baseline-results", default=str(Path("results") / "counterfactual_2x2.csv"))
    parser.add_argument("--results-out", default=str(Path("results") / "aif_plus_results.csv"))
    parser.add_argument("--window-errors-out", default=str(Path("results") / "aif_plus_window_errors.csv"))
    parser.add_argument("--arg-out", default=str(Path("results") / "aif_plus_artifact_reliance_gap.csv"))
    parser.add_argument("--wgr-out", default=str(Path("results") / "aif_plus_worst_group_risk.csv"))
    parser.add_argument("--ri-out", default=str(Path("results") / "aif_plus_ranking_instability.csv"))
    parser.add_argument("--report-out", default=str(Path("reports") / "aif_plus_summary.md"))
    return parser.parse_args()


def orthogonality_loss(z_clean: torch.Tensor, z_art: torch.Tensor) -> torch.Tensor:
    if z_clean.shape[0] <= 1:
        return (z_clean.sum() + z_art.sum()) * 0.0
    z_clean_centered = z_clean - z_clean.mean(dim=0, keepdim=True)
    z_art_centered = z_art - z_art.mean(dim=0, keepdim=True)
    z_clean_norm = F.normalize(z_clean_centered, dim=-1)
    z_art_norm = F.normalize(z_art_centered, dim=-1)
    cross = z_clean_norm.transpose(0, 1) @ z_art_norm / float(z_clean.shape[0])
    return cross.pow(2).mean()


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * max(float(warmup_ratio), 0.0)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def evaluate_aif_plus(
    model: nn.Module,
    dataset_bundle: Any,
    events_lookup: dict[str, dict[str, Any]],
    eval_rows: pd.DataFrame,
    clean_view_name: str,
    runtime_cfg: dict[str, Any],
    metadata_context: dict[str, Any],
    group_map: dict[str, int],
    artifact_map: dict[str, int],
    phase_map: dict[str, int],
    support_id_lookup: dict[str, int],
    horizon_id: int,
    force_intervened_input: bool,
    split_name: str,
    setting_meta: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame]:
    if eval_rows.empty:
        return {"mae": float("nan"), "mse": float("nan"), "smape": float("nan")}, pd.DataFrame()
    device = next(model.parameters()).device
    amp_enabled = bool(runtime_cfg.get("amp", True)) and device.type == "cuda"
    pin_memory = bool(runtime_cfg.get("pin_memory", True)) and device.type == "cuda"
    loader = build_loader(
        AIFPlusWindowDataset(
            rows=eval_rows,
            dataset_bundle=dataset_bundle,
            events_lookup=events_lookup,
            clean_view_name=clean_view_name,
            group_map=group_map,
            artifact_map=artifact_map,
            phase_map=phase_map,
            dataset_id=int(metadata_context["dataset_id"]),
            support_id_lookup=support_id_lookup,
            horizon_id=int(horizon_id),
            split_name=split_name,
            force_intervened_input=force_intervened_input,
        ),
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
    error_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            x_raw = batch["x_raw"].to(device, non_blocking=True)
            x_masked = batch["x_masked"].to(device, non_blocking=True)
            uncertainty = batch["uncertainty"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            metadata_num = batch["metadata_num"].to(device, non_blocking=True)
            dataset_id = batch["dataset_id"].to(device, non_blocking=True)
            support_id = batch["support_id"].to(device, non_blocking=True)
            horizon_id_tensor = batch["horizon_id"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model.predict(
                    x_raw=x_raw,
                    x_masked=x_masked,
                    uncertainty=uncertainty,
                    metadata_num=metadata_num,
                    dataset_id=dataset_id,
                    support_id=support_id,
                    horizon_id=horizon_id_tensor,
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
                error_rows.append(
                    {
                        **setting_meta,
                        "window_id": str(batch["window_id"][idx]),
                        "group_key": str(batch["group_key"][idx]),
                        "phase_group": str(batch["phase_group"][idx]),
                        "artifact_group_major": str(batch["artifact_group_major"][idx]),
                        "is_flagged": int(batch["flagged_mask"][idx]),
                        "has_input_intervention": int(batch["has_input_intervention"][idx]),
                        "strict_target_clean": int(batch["strict_target_clean"][idx]),
                        "subset_name": str(batch["subset_name"][idx]),
                        "mae": float(mae_vec[idx].detach().cpu().item()),
                        "mse": float(mse_vec[idx].detach().cpu().item()),
                        "smape": float(smape_vec[idx].detach().cpu().item()),
                    }
                )

    if total_count == 0:
        return {"mae": float("nan"), "mse": float("nan"), "smape": float("nan")}, pd.DataFrame(error_rows)
    metrics = {
        "mae": float(total_mae / total_count),
        "mse": float(total_mse / total_count),
        "smape": float(total_smape / total_count),
    }
    return metrics, pd.DataFrame(error_rows)


def select_validation_rows(view_df: pd.DataFrame, clean_view_name: str, max_rows: int | None) -> pd.DataFrame:
    clean_val = select_view_rows(view_df, split_name="val", view_name=clean_view_name, max_rows=max_rows)
    if not clean_val.empty:
        return clean_val
    raw_val = select_view_rows(view_df, split_name="val", view_name="raw", max_rows=max_rows)
    if not raw_val.empty:
        return raw_val
    return select_view_rows(view_df, split_name="train", view_name=clean_view_name, max_rows=max_rows)


def load_support_status_lookup(summary_path: Path, dataset_name: str, lookback: int, horizon: int) -> dict[str, str]:
    if not summary_path.exists():
        return {"train": "unknown", "val": "unknown", "test": "unknown"}
    df = pd.read_csv(summary_path)
    subset = df.loc[
        (df["dataset_name"] == dataset_name)
        & (df["lookback"] == int(lookback))
        & (df["horizon"] == int(horizon))
    ].copy()
    lookup = {"train": "unknown", "val": "unknown", "test": "unknown"}
    for row in subset.itertuples(index=False):
        lookup[str(row.split_name)] = str(row.view_status)
    return lookup


def load_support_vocab(summary_path: Path) -> dict[str, int]:
    if not summary_path.exists():
        return build_lookup(["unknown"])
    df = pd.read_csv(summary_path)
    values = ["unknown"] + df["view_status"].fillna("unknown").astype(str).tolist()
    return build_lookup(values)


def fit_stage(
    stage_name: str,
    model: AIFPlus,
    ema: ModelEMA | None,
    train_loader: DataLoader[Any],
    val_rows: pd.DataFrame,
    dataset_bundle: Any,
    events_lookup: dict[str, dict[str, Any]],
    clean_view_name: str,
    runtime_cfg: dict[str, Any],
    stage_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    metadata_context: dict[str, Any],
    group_map: dict[str, int],
    artifact_map: dict[str, int],
    phase_map: dict[str, int],
    support_id_lookup: dict[str, int],
    horizon_id: int,
    seed: int,
    log_prefix: str,
) -> tuple[float, float, int]:
    epochs = int(stage_cfg.get("epochs", 0))
    if epochs <= 0 or len(train_loader.dataset) == 0:
        return float("inf"), 0

    device = next(model.parameters()).device
    amp_enabled = bool(runtime_cfg.get("amp", True)) and device.type == "cuda"
    scaler = build_grad_scaler(device, enabled=amp_enabled)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage_cfg.get("lr", runtime_cfg.get("lr", 2e-4))),
        weight_decay=float(runtime_cfg.get("weight_decay", 0.05)),
        betas=parse_optimizer_betas(runtime_cfg),
    )
    total_steps = max(1, epochs * len(train_loader))
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=float(runtime_cfg.get("warmup_ratio", 0.05)),
    )
    grad_clip = float(runtime_cfg.get("grad_clip", 1.0))
    patience = int(stage_cfg.get("patience", runtime_cfg.get("patience", 8)))

    best_state = copy.deepcopy((ema.shadow if ema is not None else model).state_dict())
    best_val_mae = float("inf")
    best_val_mse = float("inf")
    epochs_ran = 0
    patience_counter = 0
    set_random_seed(seed)
    log_progress(
        f"{log_prefix} stage={stage_name} start epochs={epochs} lr={float(stage_cfg.get('lr', runtime_cfg.get('lr', 2e-4))):.2e} "
        f"train_rows={len(train_loader.dataset)} val_rows={len(val_rows)}"
    )

    lambda_pair = float(loss_cfg.get("alpha_pair", 0.2))
    lambda_rec = float(loss_cfg.get("gamma_rec", 0.2))
    lambda_cvar = float(loss_cfg.get("delta_cvar", 0.15))
    lambda_orth = float(loss_cfg.get("lambda_orth", 1e-3))
    lambda_router_balance = float(loss_cfg.get("lambda_router_balance", 0.0))
    lambda_artifact_adv = float(loss_cfg.get("lambda_artifact_adv", loss_cfg.get("lambda_adv", 0.05)))
    lambda_phase_adv = float(loss_cfg.get("lambda_phase_adv", loss_cfg.get("lambda_adv", 0.05)))
    lambda_artifact_aux = float(loss_cfg.get("lambda_artifact_aux", loss_cfg.get("lambda_adv", 0.05)))
    lambda_phase_aux = float(loss_cfg.get("lambda_phase_aux", loss_cfg.get("lambda_adv", 0.05)))
    grl_alpha_final = float(loss_cfg.get("grl_alpha_final", 1.0))
    grl_warmup_ratio = float(loss_cfg.get("grl_warmup_ratio", 0.0))
    tail_frac = float(loss_cfg.get("cvar_tail_frac", 0.2))
    global_step = 0
    warmup_steps = max(1, int(total_steps * max(grl_warmup_ratio, 0.0)))

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            global_step += 1
            if bool(stage_cfg.get("use_adv", False)):
                if grl_alpha_final <= 0.0:
                    model.grl.alpha = 0.0
                elif grl_warmup_ratio > 0.0:
                    model.grl.alpha = grl_alpha_final * min(1.0, float(global_step) / float(warmup_steps))
                else:
                    model.grl.alpha = grl_alpha_final
            else:
                model.grl.alpha = 0.0

            x_raw = batch["x_raw"].to(device, non_blocking=True)
            x_masked = batch["x_masked"].to(device, non_blocking=True)
            x_cf = batch["x_cf"].to(device, non_blocking=True)
            uncertainty = batch["uncertainty"].to(device, non_blocking=True)
            rec_mask = batch["rec_mask"].to(device, non_blocking=True)
            rec_target = batch["rec_target"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            metadata_num = batch["metadata_num"].to(device, non_blocking=True)
            dataset_id = batch["dataset_id"].to(device, non_blocking=True)
            support_id = batch["support_id"].to(device, non_blocking=True)
            horizon_tensor = batch["horizon_id"].to(device, non_blocking=True)
            artifact_id = batch["artifact_id"].to(device, non_blocking=True)
            phase_id = batch["phase_id"].to(device, non_blocking=True)
            group_id = batch["group_id"].to(device, non_blocking=True)
            pair_available = batch["pair_available"].to(device, non_blocking=True) > 0.5

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(
                    x_raw=x_raw,
                    x_masked=x_masked,
                    uncertainty=uncertainty,
                    metadata_num=metadata_num,
                    dataset_id=dataset_id,
                    support_id=support_id,
                    horizon_id=horizon_tensor,
                )
                pred = outputs["pred"]
                per_sample_mse = F.mse_loss(pred, y, reduction="none").mean(dim=(1, 2))
                loss = per_sample_mse.mean() if bool(stage_cfg.get("use_forecast_loss", True)) else pred.new_tensor(0.0)

                rec_loss = masked_mse_loss(outputs["reconstruction"], rec_target, rec_mask)
                if bool(stage_cfg.get("use_reconstruction", True)):
                    loss = loss + lambda_rec * rec_loss

                balance_loss = pred.new_tensor(0.0)
                if lambda_router_balance > 0.0:
                    balance_loss = router_balance_loss(outputs["router_weights"])
                    loss = loss + lambda_router_balance * balance_loss

                pair_loss = pred.new_tensor(0.0)
                if bool(stage_cfg.get("use_pair", False)) and bool(pair_available.any()):
                    cf_outputs = model(
                        x_raw=x_cf,
                        x_masked=x_cf,
                        uncertainty=torch.zeros_like(uncertainty),
                        metadata_num=metadata_num,
                        dataset_id=dataset_id,
                        support_id=support_id,
                        horizon_id=horizon_tensor,
                    )
                    pair_loss = ((pred[pair_available] - cf_outputs["pred"][pair_available]) ** 2).mean()
                    loss = loss + lambda_pair * pair_loss

                adv_loss = pred.new_tensor(0.0)
                if bool(stage_cfg.get("use_adv", False)):
                    adv_loss = (
                        lambda_artifact_adv * F.cross_entropy(outputs["artifact_logits_clean"], artifact_id)
                        + lambda_phase_adv * F.cross_entropy(outputs["phase_logits_clean"], phase_id)
                        + lambda_artifact_aux * F.cross_entropy(outputs["artifact_logits_art"], artifact_id)
                        + lambda_phase_aux * F.cross_entropy(outputs["phase_logits_art"], phase_id)
                    )
                    loss = loss + adv_loss

                cvar_loss = pred.new_tensor(0.0)
                if bool(stage_cfg.get("use_cvar", False)):
                    cvar_loss = group_tail_mean(per_sample_mse, group_id, tail_frac=tail_frac, round_up=True)
                    loss = loss + lambda_cvar * cvar_loss

                orth_loss = pred.new_tensor(0.0)
                if bool(stage_cfg.get("use_orth", False)):
                    orth_loss = orthogonality_loss(outputs["z_clean"], outputs["z_art"])
                    loss = loss + lambda_orth * orth_loss

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
        val_metrics, _ = evaluate_aif_plus(
            model=eval_model,
            dataset_bundle=dataset_bundle,
            events_lookup=events_lookup,
            eval_rows=val_rows,
            clean_view_name=clean_view_name,
            runtime_cfg=runtime_cfg,
            metadata_context=metadata_context,
            group_map=group_map,
            artifact_map=artifact_map,
            phase_map=phase_map,
            support_id_lookup=support_id_lookup,
            horizon_id=horizon_id,
            force_intervened_input=False,
            split_name="val",
            setting_meta={},
        )
        epochs_ran = epoch
        current_val_mae = float(val_metrics["mae"])
        current_val_mse = float(val_metrics["mse"])
        log_progress(
            f"{log_prefix} stage={stage_name} epoch {epoch}/{epochs} val_clean_mse={current_val_mse:.6f} "
            f"best_val_mse={min(best_val_mse, current_val_mse):.6f} val_clean_mae={current_val_mae:.6f}"
        )
        better_mse = current_val_mse + 1e-6 < best_val_mse
        tie_better_mae = abs(current_val_mse - best_val_mse) <= 1e-6 and current_val_mae + 1e-6 < best_val_mae
        if better_mse or tie_better_mae:
            best_val_mse = current_val_mse
            best_val_mae = current_val_mae
            best_state = copy.deepcopy((ema.shadow if ema is not None else model).state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if ema is not None:
        ema.shadow.load_state_dict(best_state)
        model.load_state_dict(best_state)
    else:
        model.load_state_dict(best_state)
    log_progress(
        f"{log_prefix} stage={stage_name} done epochs_ran={epochs_ran} "
        f"best_val_mse={best_val_mse:.6f} best_val_mae={best_val_mae:.6f}"
    )
    return best_val_mse, best_val_mae, epochs_ran


def build_summary_markdown(
    aif_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    arg_df: pd.DataFrame,
    wgr_df: pd.DataFrame,
    ri_df: pd.DataFrame,
) -> str:
    merged = compare_against_baseline(aif_df, baseline_df)
    lines = [
        "# AIF-Plus Summary",
        "",
        "本轮新增了 `AIF-Plus` 单模型 clean-view 训练脚本，包含 soft masking、双分支 latent split、GRL disentangle、4-expert decoder 和 Stage A/B/C 训练日程。",
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
                f"- {row.dataset_name} / L{int(row.lookback)} / H{row.horizon} / {row.eval_view_name}: "
                f"AIFPlus={row.mae:.4f}, ERM={row.baseline_mae:.4f}, delta={row.delta_mae_vs_baseline:.4f}"
            )
    else:
        lines.append("- 当前没有可与 ERM baseline 对齐的 AIF-Plus 结果。")
    lines.extend(["", "## Robustness Outputs", ""])
    if not arg_df.empty:
        for row in arg_df.sort_values("ARG_mae", ascending=False).head(6).itertuples(index=False):
            lines.append(f"- ARG {row.dataset_name} / L{int(row.lookback)} / H{row.horizon}: {row.ARG_mae:.4f}")
    if not wgr_df.empty:
        for row in wgr_df.sort_values("WGR_gap", ascending=False).head(6).itertuples(index=False):
            lines.append(
                f"- WGR {row.dataset_name} / L{int(row.lookback)} / H{row.horizon} / {row.eval_view_name}: "
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
    config = load_config(ROOT_DIR / Path(args.config))
    defaults = dict(config.get("defaults", {}))
    datasets = [canonicalize_dataset_name(str(name)) for name in defaults.get("datasets", [])]
    horizons = [int(item) for item in defaults.get("horizons", [])]
    seeds = [int(item) for item in defaults.get("seeds", [0])]
    dataset_id_map = {name: idx for idx, name in enumerate(datasets)}
    horizon_id_map = {int(value): idx for idx, value in enumerate(horizons)}
    views_dir = ROOT_DIR / Path(args.views_dir)
    registry_path = ROOT_DIR / Path(args.registry)
    events_path = ROOT_DIR / Path(args.events)
    support_summary_path = ROOT_DIR / Path(args.support_summary)
    baseline_results_path = ROOT_DIR / Path(args.baseline_results)
    support_vocab = load_support_vocab(support_summary_path)

    result_rows: list[dict[str, Any]] = []
    error_frames: list[pd.DataFrame] = []
    bundle_cache: dict[str, Any] = {}
    events_cache: dict[str, dict[str, Any]] = {}
    log_progress(f"start datasets={datasets} horizons={horizons}")

    for dataset_name in datasets:
        dataset_cfg = resolve_aif_plus_dataset_config(defaults, dataset_name)
        runtime_cfg = dict(dataset_cfg["runtime"])
        model_cfg = dict(dataset_cfg["model"])
        loss_cfg = dict(dataset_cfg["loss"])
        stage_cfg = dict(dataset_cfg["stages"])
        lookback = int(dataset_cfg["lookback"])
        collapse_aux_label_vocabs = bool(dataset_cfg["collapse_aux_label_vocabs"])
        bundle_cache[dataset_name] = load_dataset_bundle(dataset_name, registry_path=registry_path)
        events_cache[dataset_name] = load_events_lookup(events_path=events_path, dataset_name=dataset_name)
        clean_view_name = resolve_clean_view_name(config, dataset_name)
        for horizon in horizons:
            log_progress(f"prepare dataset={dataset_name} L{lookback} H{horizon}")
            view_df = load_view_frame(views_dir, dataset_name, lookback=lookback, horizon=horizon)
            train_clean_rows = select_view_rows(
                view_df,
                split_name="train",
                view_name=clean_view_name,
                max_rows=runtime_cfg.get("max_train_windows"),
            )
            train_raw_rows = select_view_rows(
                view_df,
                split_name="train",
                view_name="raw",
                max_rows=runtime_cfg.get("max_train_windows"),
            )
            val_rows = select_validation_rows(
                view_df=view_df,
                clean_view_name=clean_view_name,
                max_rows=runtime_cfg.get("max_val_windows"),
            )
            if train_clean_rows.empty or train_raw_rows.empty or val_rows.empty:
                log_progress(f"skip dataset={dataset_name} L{lookback} H{horizon} due to empty train/val rows")
                continue

            support_status_lookup = load_support_status_lookup(
                summary_path=support_summary_path,
                dataset_name=dataset_name,
                lookback=lookback,
                horizon=horizon,
            )
            support_id_lookup = {split_name: support_vocab.get(status, 0) for split_name, status in support_status_lookup.items()}
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
            group_map = build_group_map(train_raw_rows)
            metadata_context = {
                "dataset_id": dataset_id_map[dataset_name],
                "horizon_id": horizon_id_map[horizon],
                "support_train": support_id_lookup.get("train", 0),
                "support_val": support_id_lookup.get("val", 0),
                "support_test": support_id_lookup.get("test", 0),
            }

            for seed in seeds:
                set_random_seed(seed)
                model = AIFPlus(
                    AIFPlusConfig(
                        seq_len=lookback,
                        pred_len=horizon,
                        enc_in=bundle_cache[dataset_name].n_vars,
                        metadata_num_dim=len(METADATA_NUMERIC_COLUMNS),
                        artifact_vocab_size=len(artifact_map),
                        phase_vocab_size=len(phase_map),
                        dataset_vocab_size=max(len(dataset_id_map), 1),
                        support_vocab_size=max(len(support_vocab), 1),
                        horizon_vocab_size=max(len(horizon_id_map), 1),
                        d_model=int(model_cfg.get("d_model", 256)),
                        latent_dim=int(model_cfg.get("latent_dim", 256)),
                        expert_hidden=int(model_cfg.get("expert_hidden", 384)),
                        head_rank=int(model_cfg.get("head_rank", 32)),
                        patch_len=int(model_cfg.get("patch_len", 8)),
                        patch_stride=int(model_cfg.get("patch_stride", 4)),
                        n_blocks=int(model_cfg.get("n_blocks", 4)),
                        n_heads=int(model_cfg.get("n_heads", 8)),
                        ffn_ratio=int(model_cfg.get("ffn_ratio", 4)),
                        dropout=float(model_cfg.get("dropout", 0.1)),
                        stochastic_depth=float(model_cfg.get("stochastic_depth", 0.1)),
                        num_experts=int(model_cfg.get("num_experts", 4)),
                        epsilon_nuisance=float(model_cfg.get("epsilon_nuisance", 0.05)),
                        use_diff_branch=bool(model_cfg.get("use_diff_branch", dataset_name in {"solar_AL", "weather"})),
                    )
                )
                device = resolve_device(str(runtime_cfg.get("device", "auto")))
                model.to(device)
                ema = ModelEMA(model, decay=float(runtime_cfg.get("ema_decay", 0.999))) if bool(runtime_cfg.get("use_ema", True)) else None
                pin_memory = bool(runtime_cfg.get("pin_memory", True)) and device.type == "cuda"

                stage_a_loader = build_loader(
                    AIFPlusWindowDataset(
                        rows=train_raw_rows,
                        dataset_bundle=bundle_cache[dataset_name],
                        events_lookup=events_cache[dataset_name],
                        clean_view_name=clean_view_name,
                        group_map=group_map,
                        artifact_map=artifact_map,
                        phase_map=phase_map,
                        dataset_id=dataset_id_map[dataset_name],
                        support_id_lookup=support_id_lookup,
                        horizon_id=horizon_id_map[horizon],
                        split_name="train",
                        synthetic_pretrain=True,
                        synthetic_prob=float(stage_cfg.get("stage_a", {}).get("synthetic_prob", 0.7)),
                        synthetic_seed=seed + horizon,
                    ),
                    batch_size=int(runtime_cfg.get("batch_size", 64)),
                    shuffle=True,
                    num_workers=int(runtime_cfg.get("num_workers", 0)),
                    pin_memory=pin_memory,
                )
                stage_b_loader = build_loader(
                    AIFPlusWindowDataset(
                        rows=train_clean_rows,
                        dataset_bundle=bundle_cache[dataset_name],
                        events_lookup=events_cache[dataset_name],
                        clean_view_name=clean_view_name,
                        group_map=group_map,
                        artifact_map=artifact_map,
                        phase_map=phase_map,
                        dataset_id=dataset_id_map[dataset_name],
                        support_id_lookup=support_id_lookup,
                        horizon_id=horizon_id_map[horizon],
                        split_name="train",
                    ),
                    batch_size=int(runtime_cfg.get("batch_size", 64)),
                    shuffle=True,
                    num_workers=int(runtime_cfg.get("num_workers", 0)),
                    pin_memory=pin_memory,
                )
                stage_c_loader = build_loader(
                    AIFPlusWindowDataset(
                        rows=train_raw_rows,
                        dataset_bundle=bundle_cache[dataset_name],
                        events_lookup=events_cache[dataset_name],
                        clean_view_name=clean_view_name,
                        group_map=group_map,
                        artifact_map=artifact_map,
                        phase_map=phase_map,
                        dataset_id=dataset_id_map[dataset_name],
                        support_id_lookup=support_id_lookup,
                        horizon_id=horizon_id_map[horizon],
                        split_name="train",
                    ),
                    batch_size=int(runtime_cfg.get("batch_size", 64)),
                    shuffle=True,
                    num_workers=int(runtime_cfg.get("num_workers", 0)),
                    pin_memory=pin_memory,
                )

                log_prefix = f"{dataset_name}/L{lookback}/H{horizon}/seed{seed}"
                _, _, stage_a_epochs = fit_stage(
                    stage_name="stage_a",
                    model=model,
                    ema=ema,
                    train_loader=stage_a_loader,
                    val_rows=val_rows,
                    dataset_bundle=bundle_cache[dataset_name],
                    events_lookup=events_cache[dataset_name],
                    clean_view_name=clean_view_name,
                    runtime_cfg=runtime_cfg,
                    stage_cfg=dict(stage_cfg.get("stage_a", {})),
                    loss_cfg=loss_cfg,
                    metadata_context=metadata_context,
                    group_map=group_map,
                    artifact_map=artifact_map,
                    phase_map=phase_map,
                    support_id_lookup=support_id_lookup,
                    horizon_id=horizon_id_map[horizon],
                    seed=seed,
                    log_prefix=log_prefix,
                )
                best_val_mse, best_val_mae, stage_b_epochs = fit_stage(
                    stage_name="stage_b",
                    model=model,
                    ema=ema,
                    train_loader=stage_b_loader,
                    val_rows=val_rows,
                    dataset_bundle=bundle_cache[dataset_name],
                    events_lookup=events_cache[dataset_name],
                    clean_view_name=clean_view_name,
                    runtime_cfg=runtime_cfg,
                    stage_cfg=dict(stage_cfg.get("stage_b", {})),
                    loss_cfg=loss_cfg,
                    metadata_context=metadata_context,
                    group_map=group_map,
                    artifact_map=artifact_map,
                    phase_map=phase_map,
                    support_id_lookup=support_id_lookup,
                    horizon_id=horizon_id_map[horizon],
                    seed=seed,
                    log_prefix=log_prefix,
                )
                best_val_mse, best_val_mae, stage_c_epochs = fit_stage(
                    stage_name="stage_c",
                    model=model,
                    ema=ema,
                    train_loader=stage_c_loader,
                    val_rows=val_rows,
                    dataset_bundle=bundle_cache[dataset_name],
                    events_lookup=events_cache[dataset_name],
                    clean_view_name=clean_view_name,
                    runtime_cfg=runtime_cfg,
                    stage_cfg=dict(stage_cfg.get("stage_c", {})),
                    loss_cfg=loss_cfg,
                    metadata_context=metadata_context,
                    group_map=group_map,
                    artifact_map=artifact_map,
                    phase_map=phase_map,
                    support_id_lookup=support_id_lookup,
                    horizon_id=horizon_id_map[horizon],
                    seed=seed,
                    log_prefix=log_prefix,
                )

                eval_model = ema.shadow if ema is not None else model
                for eval_view_name, force_intervened_input in [
                    ("raw", False),
                    (clean_view_name, False),
                    ("intervened", True),
                ]:
                    eval_rows = select_view_rows(
                        view_df,
                        split_name="test",
                        view_name=eval_view_name,
                        max_rows=runtime_cfg.get("max_test_windows"),
                    )
                    if eval_rows.empty:
                        continue
                    metrics, errors = evaluate_aif_plus(
                        model=eval_model,
                        dataset_bundle=bundle_cache[dataset_name],
                        events_lookup=events_cache[dataset_name],
                        eval_rows=eval_rows,
                        clean_view_name=clean_view_name,
                        runtime_cfg=runtime_cfg,
                        metadata_context=metadata_context,
                        group_map=group_map,
                        artifact_map=artifact_map,
                        phase_map=phase_map,
                        support_id_lookup=support_id_lookup,
                        horizon_id=horizon_id_map[horizon],
                        force_intervened_input=force_intervened_input,
                        split_name="test",
                        setting_meta={
                            "dataset_name": dataset_name,
                            "backbone": "AIFPlus",
                            "lookback": lookback,
                            "horizon": horizon,
                            "train_view_name": "aif_plus",
                            "eval_view_name": eval_view_name,
                            "hyperparam_source_kind": dataset_cfg["hyperparam_source_kind"],
                            "hyperparam_source_url": dataset_cfg["hyperparam_source_url"],
                            "hyperparam_source_note": dataset_cfg["hyperparam_source_note"],
                        },
                    )
                    result_rows.append(
                        {
                            "dataset_name": dataset_name,
                            "backbone": "AIFPlus",
                            "lookback": lookback,
                            "horizon": horizon,
                            "train_view_name": "aif_plus",
                            "eval_view_name": eval_view_name,
                            "seed": seed,
                            "hyperparam_source_kind": dataset_cfg["hyperparam_source_kind"],
                            "hyperparam_source_url": dataset_cfg["hyperparam_source_url"],
                            "hyperparam_source_note": dataset_cfg["hyperparam_source_note"],
                            "n_train_windows": int(len(train_clean_rows)),
                            "n_eval_windows": int(len(eval_rows)),
                            "best_val_mae": round(float(best_val_mae), 6),
                            "best_val_mse": round(float(best_val_mse), 6),
                            "stage_a_epochs_ran": int(stage_a_epochs),
                            "stage_b_epochs_ran": int(stage_b_epochs),
                            "stage_c_epochs_ran": int(stage_c_epochs),
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
    ri_df = compute_aif_ri_table(results_df)

    results_out = ROOT_DIR / Path(args.results_out)
    window_errors_out = ROOT_DIR / Path(args.window_errors_out)
    arg_out = ROOT_DIR / Path(args.arg_out)
    wgr_out = ROOT_DIR / Path(args.wgr_out)
    ri_out = ROOT_DIR / Path(args.ri_out)
    report_out = ROOT_DIR / Path(args.report_out)
    for path in [results_out, window_errors_out, arg_out, wgr_out, ri_out, report_out]:
        path.parent.mkdir(parents=True, exist_ok=True)
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
