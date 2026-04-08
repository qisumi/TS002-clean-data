from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib.util
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
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
AIFPlusLoss = _AIFPLUS_MODULE.AIFPlusLoss


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

WINDOW_ERROR_MINIMAL_FIELDS = [
    "dataset_name",
    "backbone",
    "lookback",
    "horizon",
    "train_view_name",
    "eval_view_name",
    "seed",
    "checkpoint_variant",
    "window_id",
    "group_key",
    "mae",
    "mse",
    "smape",
    "is_valid_metric",
    "has_input_intervention",
    "strict_target_clean",
]

WINDOW_ERROR_RICH_FIELDS = [
    "phase_group",
    "artifact_group_major",
    "is_flagged",
    "subset_name",
]


def parse_optimizer_betas(runtime_cfg: dict[str, Any]) -> tuple[float, float]:
    raw_betas = runtime_cfg.get("betas", (0.9, 0.999))
    if isinstance(raw_betas, (list, tuple)) and len(raw_betas) == 2:
        return float(raw_betas[0]), float(raw_betas[1])
    return 0.9, 0.999


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * max(float(warmup_ratio), 0.0)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def unwrap_model(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.DataParallel) else module


def clone_tensor_state_dict(state_dict: dict[str, torch.Tensor] | dict[str, Any]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return clone_tensor_state_dict(unwrap_model(module).state_dict())


def load_model_state(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    unwrap_model(module).load_state_dict(state_dict)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float, device: str = "same") -> None:
        self.decay = float(decay)
        base_model = unwrap_model(model)
        if hasattr(base_model, "config"):
            self.shadow = type(base_model)(base_model.config).eval()
            self.shadow.load_state_dict(clone_tensor_state_dict(base_model.state_dict()))
        else:
            self.shadow = copy.deepcopy(base_model).eval()
        target_device = str(device).strip().lower()
        if target_device in {"", "same", "model"}:
            shadow_device = next(base_model.parameters()).device
        else:
            shadow_device = torch.device(target_device)
        self._shadow_device = shadow_device
        self.shadow.to(shadow_device)
        for param in self.shadow.parameters():
            param.requires_grad_(False)
        self._state_names = tuple(self.shadow.state_dict().keys())
        self._update_scratch_state: dict[str, torch.Tensor] = {}
        self._backup_state: dict[str, torch.Tensor] = (
            self._update_scratch_state if self._shadow_device.type == "cpu" else {}
        )
        self._backup_ready = False

    @staticmethod
    def _ensure_tensor_slot(
        storage: dict[str, torch.Tensor],
        name: str,
        reference: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        scratch = storage.get(name)
        if scratch is None or scratch.shape != reference.shape or scratch.dtype != reference.dtype or scratch.device != device:
            scratch = torch.empty_like(reference, device=device)
            storage[name] = scratch
        return scratch

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self._backup_ready:
            raise RuntimeError("EMA update called while a live model backup is still active")
        model_state = unwrap_model(model).state_dict()
        shadow_state = self.shadow.state_dict()
        for name in self._state_names:
            value = shadow_state[name]
            source = model_state[name].detach()
            if source.device != value.device:
                source_shadow = self._ensure_tensor_slot(
                    self._update_scratch_state,
                    name,
                    value,
                    device=value.device,
                )
                source_shadow.copy_(source, non_blocking=False)
                source = source_shadow
            if not torch.is_floating_point(source):
                value.copy_(source)
                continue
            value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return self.shadow.state_dict()

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        model_state = unwrap_model(model).state_dict()
        shadow_state = self.shadow.state_dict()
        for name in self._state_names:
            backup = self._ensure_tensor_slot(
                self._backup_state,
                name,
                shadow_state[name],
                device=torch.device("cpu"),
            )
            backup.copy_(model_state[name], non_blocking=False)
        self._backup_ready = True

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model_state = unwrap_model(model).state_dict()
        shadow_state = self.shadow.state_dict()
        for name in self._state_names:
            model_state[name].copy_(shadow_state[name], non_blocking=False)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self._backup_ready:
            raise RuntimeError("EMA restore called before store")
        model_state = unwrap_model(model).state_dict()
        for name in self._state_names:
            model_state[name].copy_(self._backup_state[name], non_blocking=False)
        self._backup_ready = False


class WindowErrorCSVWriter:
    def __init__(self, path: Path, *, rich_fields: bool = False) -> None:
        self.path = path
        self.fieldnames = list(WINDOW_ERROR_MINIMAL_FIELDS)
        if rich_fields:
            self.fieldnames.extend(WINDOW_ERROR_RICH_FIELDS)
        self._handle: Any | None = None
        self._writer: csv.DictWriter | None = None
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def __enter__(self) -> WindowErrorCSVWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        if not self._header_written:
            self._writer.writeheader()
            self._header_written = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._writer = None

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows or self._writer is None:
            return
        self._writer.writerows(rows)


def evaluate_with_ema_state(
    *,
    model: nn.Module,
    ema: ModelEMA | None,
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
    setting_meta: dict[str, Any] | None,
    window_error_path: Path | None = None,
    write_window_errors: bool = False,
    window_error_rich_fields: bool = False,
) -> dict[str, float]:
    if ema is None:
        return evaluate_aif_plus(
            model=model,
            dataset_bundle=dataset_bundle,
            events_lookup=events_lookup,
            eval_rows=eval_rows,
            clean_view_name=clean_view_name,
            runtime_cfg=runtime_cfg,
            metadata_context=metadata_context,
            group_map=group_map,
            artifact_map=artifact_map,
            phase_map=phase_map,
            support_id_lookup=support_id_lookup,
            horizon_id=horizon_id,
            force_intervened_input=force_intervened_input,
            split_name=split_name,
            setting_meta=setting_meta,
            window_error_path=window_error_path,
            write_window_errors=write_window_errors,
            window_error_rich_fields=window_error_rich_fields,
        )
    ema.store(model)
    ema.copy_to(model)
    try:
        return evaluate_aif_plus(
            model=model,
            dataset_bundle=dataset_bundle,
            events_lookup=events_lookup,
            eval_rows=eval_rows,
            clean_view_name=clean_view_name,
            runtime_cfg=runtime_cfg,
            metadata_context=metadata_context,
            group_map=group_map,
            artifact_map=artifact_map,
            phase_map=phase_map,
            support_id_lookup=support_id_lookup,
            horizon_id=horizon_id,
            force_intervened_input=force_intervened_input,
            split_name=split_name,
            setting_meta=setting_meta,
            window_error_path=window_error_path,
            write_window_errors=write_window_errors,
            window_error_rich_fields=window_error_rich_fields,
        )
    finally:
        ema.restore(model)


def compute_val_score(metrics: dict[str, float]) -> float:
    return 0.6 * float(metrics["mae"]) + 0.4 * float(metrics["mse"])


def parse_bool_arg(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"", "1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got `{value}`")


def parse_csv_values(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def resolve_cli_subset(raw_value: str | None, caster: Any) -> list[Any] | None:
    items = parse_csv_values(raw_value)
    if not items:
        return None
    return [caster(item) for item in items]


def with_run_tag(path_text: str, default_path: str, run_tag: str) -> str:
    if not run_tag or path_text != default_path:
        return path_text
    path = Path(path_text)
    return str(path.with_name(f"{path.stem}.{run_tag}{path.suffix}"))


def resolve_checkpoint_sort_key(mae: float, mse: float, epoch: int) -> tuple[float, float, int]:
    mae_rank = float(mae) if math.isfinite(mae) else float("inf")
    mse_rank = float(mse) if math.isfinite(mse) else float("inf")
    return (mae_rank, mse_rank, int(epoch))


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
        force_intervened_input: bool = False,
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
        self.force_intervened_input = bool(force_intervened_input)

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
            if str(op.get("op", "")) == "drop_window":
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
    def _safe_text(value: Any, fallback: str = "NA") -> str:
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

        x_raw = self.dataset_bundle.scaled_values[input_start : input_end + 1].copy()
        y = self.dataset_bundle.scaled_values[target_start : target_end + 1].copy()
        x_cf, rec_mask, uncertainty = self._soft_mask_from_recipe(row=row, x_raw=x_raw, input_start=input_start)

        if self.force_intervened_input:
            x_raw = x_cf.copy()
            x_masked = x_cf.copy()
            uncertainty = np.zeros_like(x_raw, dtype=np.float32)
        else:
            x_masked = (1.0 - rec_mask) * x_raw + rec_mask * x_cf

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
            "uncertainty": torch.from_numpy(uncertainty.astype(np.float32, copy=False)),
            "y": torch.from_numpy(y.astype(np.float32, copy=False)),
            "metadata_num": torch.from_numpy(metadata_num),
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.long),
            "support_id": torch.tensor(self.support_id, dtype=torch.long),
            "horizon_id": torch.tensor(self.horizon_id, dtype=torch.long),
            "artifact_id": torch.tensor(self.artifact_map.get(artifact_name, 0), dtype=torch.long),
            "phase_id": torch.tensor(self.phase_map.get(phase_name, 0), dtype=torch.long),
            "group_id": torch.tensor(self.group_map.get(self._safe_text(row.get("primary_group_key", "NA")), 0), dtype=torch.long),
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
    parser = argparse.ArgumentParser(description="Run AIF-Plus clean-first training.")
    parser.add_argument("--config", default=str(Path("configs") / "aif_plus.yaml"))
    parser.add_argument("--views-dir", default=str(Path("statistic_results") / "window_views"))
    parser.add_argument("--registry", default=str(Path("statistic_results") / "dataset_registry.csv"))
    parser.add_argument("--events", default=str(Path("statistic_results") / "final_artifact_events.csv"))
    parser.add_argument("--support-summary", default=str(Path("reports") / "clean_view_support_summary.csv"))
    parser.add_argument("--baseline-results", default=str(Path("results") / "counterfactual_2x2.csv"))
    parser.add_argument("--results-out", default=str(Path("results") / "aif_plus_results.csv"))
    parser.add_argument("--results-online-out", default="")
    parser.add_argument("--validity-out", default="")
    parser.add_argument("--window-errors-out", default=str(Path("results") / "aif_plus_window_errors.csv"))
    parser.add_argument("--arg-out", default=str(Path("results") / "aif_plus_artifact_reliance_gap.csv"))
    parser.add_argument("--wgr-out", default=str(Path("results") / "aif_plus_worst_group_risk.csv"))
    parser.add_argument("--ri-out", default=str(Path("results") / "aif_plus_ranking_instability.csv"))
    parser.add_argument("--report-out", default=str(Path("reports") / "aif_plus_summary.md"))
    parser.add_argument("--datasets", default="")
    parser.add_argument("--horizons", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--write-window-errors", type=parse_bool_arg, default=True)
    parser.add_argument("--window-error-rich-fields", type=parse_bool_arg, default=False)
    parser.add_argument("--debug-write-val-window-errors", type=parse_bool_arg, default=False)
    parser.add_argument("--checkpoint-comparison-out", default="")
    return parser.parse_args()


def resolve_experiment_meta(defaults: dict[str, Any]) -> dict[str, str]:
    display_name = str(defaults.get("experiment_name", "AIF-Plus")).strip() or "AIF-Plus"
    backbone_name = str(defaults.get("backbone_name", "AIFPlus")).strip() or "AIFPlus"
    train_view_name = str(defaults.get("train_view_name", "aif_plus")).strip() or "aif_plus"
    summary_title = str(defaults.get("summary_title", f"{display_name} Summary")).strip() or f"{display_name} Summary"
    summary_note = str(defaults.get("summary_note", "")).strip()
    return {
        "display_name": display_name,
        "backbone_name": backbone_name,
        "train_view_name": train_view_name,
        "summary_title": summary_title,
        "summary_note": summary_note,
    }


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
    setting_meta: dict[str, Any] | None,
    *,
    window_error_path: Path | None = None,
    write_window_errors: bool = False,
    window_error_rich_fields: bool = False,
) -> dict[str, float]:
    if eval_rows.empty:
        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "smape": float("nan"),
            "n_total_windows": 0,
            "n_valid_windows": 0,
            "n_invalid_windows": 0,
            "invalid_ratio": float("nan"),
        }
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
    total_windows = 0
    total_valid_windows = 0
    total_invalid_windows = 0
    writer: WindowErrorCSVWriter | None = None
    if write_window_errors and window_error_path is not None:
        writer = WindowErrorCSVWriter(window_error_path, rich_fields=window_error_rich_fields)

    try:
        with (writer or nullcontext()) as active_writer:
            with torch.no_grad():
                for batch in loader:
                    x_raw = batch["x_raw"].to(device, non_blocking=True)
                    x_masked = batch["x_masked"].to(device, non_blocking=True)
                    uncertainty = batch["uncertainty"].to(device, non_blocking=True)
                    y = batch["y"].to(device, non_blocking=True)
                    horizon_id_tensor = batch["horizon_id"].to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, enabled=amp_enabled):
                        pred = model.predict(
                            x_raw=x_raw,
                            x_masked=x_masked,
                            uncertainty=uncertainty,
                            horizon_id=horizon_id_tensor,
                        )
                    mae_vec = (pred - y).abs().mean(dim=(1, 2))
                    mse_vec = ((pred - y) ** 2).mean(dim=(1, 2))
                    smape_vec = (200.0 * (pred - y).abs() / (pred.abs() + y.abs() + 1e-6)).mean(dim=(1, 2))
                    valid_mask = torch.isfinite(mae_vec) & torch.isfinite(mse_vec) & torch.isfinite(smape_vec)
                    batch_size = int(x_raw.shape[0])
                    batch_valid = int(valid_mask.sum().item())
                    batch_invalid = batch_size - batch_valid
                    total_windows += batch_size
                    total_valid_windows += batch_valid
                    total_invalid_windows += batch_invalid

                    if batch_valid > 0:
                        total_mae += float(mae_vec[valid_mask].sum().item())
                        total_mse += float(mse_vec[valid_mask].sum().item())
                        total_smape += float(smape_vec[valid_mask].sum().item())

                    if active_writer is not None:
                        mae_cpu = mae_vec.detach().cpu().tolist()
                        mse_cpu = mse_vec.detach().cpu().tolist()
                        smape_cpu = smape_vec.detach().cpu().tolist()
                        valid_cpu = valid_mask.detach().cpu().tolist()
                        error_meta = dict(setting_meta or {})
                        batch_rows: list[dict[str, Any]] = []
                        for idx in range(batch_size):
                            row = {
                                **error_meta,
                                "window_id": str(batch["window_id"][idx]),
                                "group_key": str(batch["group_key"][idx]),
                                "mae": float(mae_cpu[idx]),
                                "mse": float(mse_cpu[idx]),
                                "smape": float(smape_cpu[idx]),
                                "is_valid_metric": int(bool(valid_cpu[idx])),
                                "has_input_intervention": int(batch["has_input_intervention"][idx]),
                                "strict_target_clean": int(batch["strict_target_clean"][idx]),
                            }
                            if window_error_rich_fields:
                                row.update(
                                    {
                                        "phase_group": str(batch["phase_group"][idx]),
                                        "artifact_group_major": str(batch["artifact_group_major"][idx]),
                                        "is_flagged": int(batch["flagged_mask"][idx]),
                                        "subset_name": str(batch["subset_name"][idx]),
                                    }
                                )
                            batch_rows.append(row)
                        active_writer.write_rows(batch_rows)
    finally:
        del writer

    invalid_ratio = (
        float(total_invalid_windows) / float(total_windows)
        if total_windows > 0
        else float("nan")
    )
    if total_valid_windows <= 0:
        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "smape": float("nan"),
            "n_total_windows": int(total_windows),
            "n_valid_windows": 0,
            "n_invalid_windows": int(total_invalid_windows),
            "invalid_ratio": invalid_ratio,
        }
    return {
        "mae": float(total_mae / total_valid_windows),
        "mse": float(total_mse / total_valid_windows),
        "smape": float(total_smape / total_valid_windows),
        "n_total_windows": int(total_windows),
        "n_valid_windows": int(total_valid_windows),
        "n_invalid_windows": int(total_invalid_windows),
        "invalid_ratio": invalid_ratio,
    }


def select_validation_rows(
    view_df: pd.DataFrame,
    clean_view_name: str,
    max_rows: int | None,
    min_clean_val_windows: int = 1,
) -> pd.DataFrame:
    clean_val = select_view_rows(view_df, split_name="val", view_name=clean_view_name, max_rows=max_rows)
    if len(clean_val) >= max(int(min_clean_val_windows), 1):
        return clean_val
    raw_val = select_view_rows(view_df, split_name="val", view_name="raw", max_rows=max_rows)
    if not raw_val.empty:
        return raw_val
    if not clean_val.empty:
        return clean_val
    return select_view_rows(view_df, split_name="train", view_name=clean_view_name, max_rows=max_rows)


def load_support_status_lookup(summary_path: Path, dataset_name: str, lookback: int, horizon: int) -> dict[str, str]:
    if not summary_path.exists():
        return {"train": "unknown", "val": "unknown", "test": "unknown"}
    df = pd.read_csv(summary_path, low_memory=False)
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
    df = pd.read_csv(summary_path, low_memory=False)
    values = ["unknown"] + df["view_status"].fillna("unknown").astype(str).tolist()
    return build_lookup(values)


def resolve_periods(model_cfg: dict[str, Any]) -> tuple[int, ...]:
    raw_periods = model_cfg.get("periods")
    if isinstance(raw_periods, (list, tuple)) and raw_periods:
        return tuple(max(1, int(item)) for item in raw_periods)
    query_period = max(1, int(model_cfg.get("query_period", 24)))
    return (query_period, query_period * 2, query_period * 4)


def resolve_queries_per_period(model_cfg: dict[str, Any], periods: tuple[int, ...]) -> int:
    if "queries_per_period" in model_cfg:
        return max(1, int(model_cfg["queries_per_period"]))
    if "num_queries" in model_cfg:
        total_queries = max(1, int(model_cfg["num_queries"]))
        return max(1, int(math.ceil(float(total_queries) / float(max(len(periods), 1)))))
    return 2


def build_model_config(
    *,
    model_cfg: dict[str, Any],
    lookback: int,
    horizon: int,
    n_vars: int,
    horizon_vocab_size: int,
) -> AIFPlusConfig:
    periods = resolve_periods(model_cfg)
    patch_len_small = int(model_cfg.get("patch_len_small", model_cfg.get("patch_len", 8)))
    patch_stride_small = int(model_cfg.get("patch_stride_small", model_cfg.get("patch_stride", 4)))
    patch_len_large = int(model_cfg.get("patch_len_large", max(patch_len_small * 2, patch_len_small + 4)))
    patch_stride_large = int(model_cfg.get("patch_stride_large", max(patch_stride_small * 2, patch_stride_small + 2)))
    return AIFPlusConfig(
        seq_len=int(lookback),
        pred_len=int(horizon),
        enc_in=int(n_vars),
        horizon_vocab_size=max(int(horizon_vocab_size), 1),
        d_model=int(model_cfg.get("d_model", 256)),
        dropout=float(model_cfg.get("dropout", 0.05)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        n_patch_layers=int(model_cfg.get("n_patch_layers", model_cfg.get("n_blocks", 2))),
        n_decoder_layers=int(model_cfg.get("n_decoder_layers", 2)),
        ffn_ratio=int(model_cfg.get("ffn_ratio", 4)),
        use_diff_branch=bool(model_cfg.get("use_diff_branch", n_vars > 8)),
        use_channel_context=bool(model_cfg.get("use_channel_context", True)),
        use_residual_branch=bool(model_cfg.get("use_residual_branch", True)),
        patch_len_small=patch_len_small,
        patch_stride_small=patch_stride_small,
        patch_len_large=patch_len_large,
        patch_stride_large=patch_stride_large,
        patch_jitter=bool(model_cfg.get("patch_jitter", True)),
        periods=periods,
        queries_per_period=resolve_queries_per_period(model_cfg, periods),
        spectral_topk=int(model_cfg.get("spectral_topk", 8)),
        residual_hidden=int(model_cfg.get("residual_hidden", model_cfg.get("resid_hidden", 32))),
        lambda_res_max=float(model_cfg.get("lambda_res_max", model_cfg.get("lambda_res_init", 0.03))),
        activation_checkpointing=bool(model_cfg.get("activation_checkpointing", True)),
        bc_chunk_size=max(int(model_cfg.get("bc_chunk_size", 1024) or 0), 0),
    )


def fit_aif_plus_best_single(
    *,
    model: nn.Module,
    ema: ModelEMA | None,
    train_loader: DataLoader[Any],
    val_rows: pd.DataFrame,
    dataset_bundle: Any,
    events_lookup: dict[str, dict[str, Any]],
    clean_view_name: str,
    runtime_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    metadata_context: dict[str, Any],
    group_map: dict[str, int],
    artifact_map: dict[str, int],
    phase_map: dict[str, int],
    support_id_lookup: dict[str, int],
    horizon_id: int,
    seed: int,
    log_prefix: str,
    val_window_error_path: Path | None = None,
    window_error_rich_fields: bool = False,
    val_setting_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    epochs = int(runtime_cfg.get("epochs", 35))
    if epochs <= 0 or len(train_loader.dataset) == 0:
        raise ValueError("AIF-Plus training requires positive epochs and non-empty train dataset")

    device = next(model.parameters()).device
    amp_enabled = bool(runtime_cfg.get("amp", True)) and device.type == "cuda"
    scaler = build_grad_scaler(device, enabled=amp_enabled)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(runtime_cfg.get("lr", 2e-4)),
        weight_decay=float(runtime_cfg.get("weight_decay", 0.05)),
        betas=parse_optimizer_betas(runtime_cfg),
    )
    total_steps = max(1, epochs * len(train_loader))
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=float(runtime_cfg.get("warmup_ratio", 0.05)),
    )
    loss_fn = AIFPlusLoss(
        mae_weight=float(loss_cfg.get("mae_weight", 0.7)),
        mse_weight=float(loss_cfg.get("mse_weight", 0.3)),
    )
    grad_clip = float(runtime_cfg.get("grad_clip", 1.0))
    patience = int(runtime_cfg.get("patience", 8))
    checkpoint_variant = "best_ema_single" if ema is not None else "best_single"

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_metrics = {
        "mae": float("nan"),
        "mse": float("nan"),
        "smape": float("nan"),
        "n_total_windows": 0,
        "n_valid_windows": 0,
        "n_invalid_windows": 0,
        "invalid_ratio": float("nan"),
    }
    best_val_score = float("nan")
    fallback_state: dict[str, torch.Tensor] | None = None
    fallback_epoch = 0
    fallback_metrics = dict(best_val_metrics)
    epochs_ran = 0
    patience_counter = 0
    set_random_seed(seed)
    log_progress(
        f"{log_prefix} fit start epochs={epochs} lr={float(runtime_cfg.get('lr', 2e-4)):.2e} "
        f"train_rows={len(train_loader.dataset)} val_rows={len(val_rows)} checkpoint={checkpoint_variant}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_mse_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            x_raw = batch["x_raw"].to(device, non_blocking=True)
            x_masked = batch["x_masked"].to(device, non_blocking=True)
            uncertainty = batch["uncertainty"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            horizon_tensor = batch["horizon_id"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(
                    x_raw=x_raw,
                    x_masked=x_masked,
                    uncertainty=uncertainty,
                    horizon_id=horizon_tensor,
                    return_aux=False,
                )
                loss_terms = loss_fn(outputs["pred"], y)
                loss = loss_terms["loss"]

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            train_loss_sum += float(loss.detach().cpu().item())
            train_mae_sum += float(loss_terms["mae"].detach().cpu().item())
            train_mse_sum += float(loss_terms["mse"].detach().cpu().item())
            train_batches += 1

        val_metrics = evaluate_with_ema_state(
            model=model,
            ema=ema,
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
            setting_meta=val_setting_meta,
            window_error_path=val_window_error_path,
            write_window_errors=val_window_error_path is not None,
            window_error_rich_fields=window_error_rich_fields,
        )
        current_val_mae = float(val_metrics["mae"])
        current_val_mse = float(val_metrics["mse"])
        current_val_score = compute_val_score(val_metrics) if math.isfinite(current_val_mae) and math.isfinite(current_val_mse) else float("nan")
        epochs_ran = epoch

        if fallback_state is None:
            fallback_state = clone_tensor_state_dict(ema.state_dict()) if ema is not None else clone_state_dict(model)
            fallback_epoch = epoch
            fallback_metrics = dict(val_metrics)

        best_score_log = (
            best_val_score
            if math.isfinite(best_val_score)
            else current_val_score
        )
        log_progress(
            f"{log_prefix} epoch {epoch}/{epochs} train_loss={train_loss_sum / max(train_batches, 1):.6f} "
            f"train_mae={train_mae_sum / max(train_batches, 1):.6f} train_mse={train_mse_sum / max(train_batches, 1):.6f} "
            f"val_mae={current_val_mae:.6f} val_mse={current_val_mse:.6f} val_score={current_val_score:.6f} "
            f"val_valid={int(val_metrics['n_valid_windows'])}/{int(val_metrics['n_total_windows'])} "
            f"invalid_ratio={float(val_metrics['invalid_ratio']):.4f} best_score={best_score_log:.6f}"
        )

        is_finite_candidate = math.isfinite(current_val_mae) and math.isfinite(current_val_mse)
        if is_finite_candidate:
            current_key = resolve_checkpoint_sort_key(current_val_mae, current_val_mse, epoch)
            best_key = resolve_checkpoint_sort_key(
                float(best_val_metrics["mae"]),
                float(best_val_metrics["mse"]),
                best_epoch if best_epoch > 0 else 10**9,
            )
            if best_state is None or current_key < best_key:
                best_state = clone_tensor_state_dict(ema.state_dict()) if ema is not None else clone_state_dict(model)
                best_epoch = epoch
                best_val_metrics = dict(val_metrics)
                best_val_score = current_val_score
                patience_counter = 0
                continue

        patience_counter += 1
        if patience_counter >= patience:
            break

    selection_status = checkpoint_variant
    if best_state is None:
        best_state = fallback_state if fallback_state is not None else clone_state_dict(model)
        best_epoch = fallback_epoch if fallback_epoch > 0 else max(epochs_ran, 1)
        best_val_metrics = dict(fallback_metrics)
        best_val_score = (
            compute_val_score(best_val_metrics)
            if math.isfinite(float(best_val_metrics["mae"])) and math.isfinite(float(best_val_metrics["mse"]))
            else float("nan")
        )
        selection_status = f"fallback_no_finite_val:{checkpoint_variant}"

    load_model_state(model, best_state)
    return {
        "epochs_ran": int(epochs_ran),
        "best_epoch": int(best_epoch),
        "best_state": best_state,
        "best_val_metrics": best_val_metrics,
        "best_val_score": float(best_val_score),
        "selection_status": selection_status,
        "checkpoint_variant": checkpoint_variant,
        "checkpoint_source_epochs": [int(best_epoch)],
    }


def build_summary_markdown(
    aif_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    arg_df: pd.DataFrame,
    wgr_df: pd.DataFrame,
    ri_df: pd.DataFrame,
    *,
    checkpoint_df: pd.DataFrame | None = None,
    summary_title: str = "AIF-Plus Summary",
    display_name: str = "AIF-Plus",
    summary_note: str = "",
) -> str:
    merged = compare_against_baseline(aif_df, baseline_df)
    intro = summary_note or (
        "本轮采用 clean-first 的 AIF-Plus：single-stage clean-first training，"
        "loss=0.7*MAE+0.3*MSE，best-single checkpoint selection。"
    )
    lines = [
        f"# {summary_title}",
        "",
        intro,
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
                f"{display_name}={row.mae:.4f}, ERM={row.baseline_mae:.4f}, delta={row.delta_mae_vs_baseline:.4f}"
            )
    else:
        lines.append("- 当前没有可与 ERM baseline 对齐的 AIF-Plus 结果。")

    if checkpoint_df is not None and not checkpoint_df.empty:
        lines.extend(["", "## Checkpoint Selection", ""])
        raw_cmp = checkpoint_df[checkpoint_df["eval_view_name"] == "raw"].copy()
        if raw_cmp.empty:
            raw_cmp = checkpoint_df.copy()
        summary = (
            raw_cmp.groupby("checkpoint_variant", dropna=False)
            .agg(
                mean_val_score=("val_score", "mean"),
                mean_test_mae=("mae", "mean"),
                n_rows=("mae", "size"),
            )
            .reset_index()
            .sort_values(["mean_test_mae", "mean_val_score"])
        )
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.checkpoint_variant}: mean_val_score={row.mean_val_score:.4f}, "
                f"mean_test_mae={row.mean_test_mae:.4f}, rows={int(row.n_rows)}"
            )

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


def sort_results_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df
    sort_columns = [
        "dataset_name",
        "backbone",
        "lookback",
        "horizon",
        "train_view_name",
        "eval_view_name",
        "seed",
    ]
    existing = [column for column in sort_columns if column in results_df.columns]
    return results_df.sort_values(existing).reset_index(drop=True)


def resolve_optional_output_path(path_text: str) -> Path | None:
    text = str(path_text).strip()
    if not text:
        return None
    return ROOT_DIR / Path(text)


def make_val_debug_window_error_path(window_errors_out: Path | None) -> Path | None:
    if window_errors_out is None:
        return None
    return window_errors_out.with_name(f"{window_errors_out.stem}_val_debug{window_errors_out.suffix}")


def resolve_results_side_outputs(
    results_out: Path | None,
    results_online_out: Path | None,
    validity_out: Path | None,
) -> tuple[Path | None, Path | None, Path | None]:
    final_results_out = results_out
    online_out = results_online_out
    diagnostics_out = validity_out

    if online_out is None and final_results_out is not None:
        stem = final_results_out.stem
        suffix = final_results_out.suffix
        if stem.endswith("_results_online"):
            online_out = final_results_out
            final_results_out = final_results_out.with_name(f"{stem[:-7]}{suffix}")
        else:
            online_out = final_results_out.with_name(f"{stem}_online{suffix}")

    if diagnostics_out is None and final_results_out is not None:
        diagnostics_out = final_results_out.with_name(f"{final_results_out.stem}_validity_diagnostics{final_results_out.suffix}")

    return final_results_out, online_out, diagnostics_out


def recompute_results_from_window_errors(
    online_results_df: pd.DataFrame,
    window_errors_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if online_results_df.empty:
        return online_results_df.copy(), pd.DataFrame()

    metrics_cols = [col for col in ["mae", "mse", "smape"] if col in window_errors_df.columns]
    if window_errors_df.empty or not metrics_cols:
        diagnostics = online_results_df.copy()
        if "n_eval_windows" in diagnostics.columns:
            diagnostics["n_total_windows"] = diagnostics["n_eval_windows"]
        diagnostics["n_valid_windows"] = diagnostics.get("n_total_windows", 0)
        diagnostics["n_invalid_windows"] = 0
        diagnostics["invalid_ratio"] = 0.0
        return sort_results_frame(online_results_df.copy()), sort_results_frame(diagnostics)

    valid_mask = np.ones(len(window_errors_df), dtype=bool)
    for col in metrics_cols:
        valid_mask &= np.isfinite(window_errors_df[col].to_numpy())
    if "is_valid_metric" in window_errors_df.columns:
        valid_mask &= window_errors_df["is_valid_metric"].fillna(0).astype(int).to_numpy().astype(bool)
    filtered = window_errors_df.loc[valid_mask].copy()

    key_cols = [
        "dataset_name",
        "backbone",
        "lookback",
        "horizon",
        "train_view_name",
        "eval_view_name",
        "seed",
        "checkpoint_variant",
    ]
    present_keys = [col for col in key_cols if col in window_errors_df.columns and col in online_results_df.columns]
    grouped_valid = (
        filtered.groupby(present_keys, dropna=False)
        .agg(
            mae=("mae", "mean"),
            mse=("mse", "mean"),
            smape=("smape", "mean"),
            n_valid_windows=("window_id", "size") if "window_id" in filtered.columns else ("mae", "size"),
        )
        .reset_index()
    )
    grouped_total = (
        window_errors_df.groupby(present_keys, dropna=False)
        .agg(
            n_total_windows=("window_id", "size") if "window_id" in window_errors_df.columns else ("mae", "size"),
        )
        .reset_index()
    )
    diagnostics = grouped_total.merge(grouped_valid, on=present_keys, how="left")
    diagnostics["n_valid_windows"] = diagnostics["n_valid_windows"].fillna(0).astype(int)
    diagnostics["n_total_windows"] = diagnostics["n_total_windows"].fillna(0).astype(int)
    diagnostics["n_invalid_windows"] = diagnostics["n_total_windows"] - diagnostics["n_valid_windows"]
    diagnostics["invalid_ratio"] = np.where(
        diagnostics["n_total_windows"] > 0,
        diagnostics["n_invalid_windows"] / diagnostics["n_total_windows"],
        np.nan,
    )

    meta_cols = [
        col
        for col in online_results_df.columns
        if col not in {"mae", "mse", "smape", "n_total_windows", "n_valid_windows", "n_invalid_windows", "invalid_ratio"}
    ]
    final_results = online_results_df[meta_cols].merge(
        diagnostics,
        on=present_keys,
        how="left",
    )
    if "n_eval_windows" in final_results.columns:
        final_results["n_eval_windows"] = final_results["n_total_windows"].fillna(final_results["n_eval_windows"])
    return sort_results_frame(final_results), sort_results_frame(diagnostics)


def main() -> None:
    args = parse_args()
    args.results_out = with_run_tag(args.results_out, str(Path("results") / "aif_plus_results.csv"), args.run_tag)
    args.window_errors_out = with_run_tag(
        args.window_errors_out,
        str(Path("results") / "aif_plus_window_errors.csv"),
        args.run_tag,
    )
    args.arg_out = with_run_tag(args.arg_out, str(Path("results") / "aif_plus_artifact_reliance_gap.csv"), args.run_tag)
    args.wgr_out = with_run_tag(args.wgr_out, str(Path("results") / "aif_plus_worst_group_risk.csv"), args.run_tag)
    args.ri_out = with_run_tag(args.ri_out, str(Path("results") / "aif_plus_ranking_instability.csv"), args.run_tag)
    args.report_out = with_run_tag(args.report_out, str(Path("reports") / "aif_plus_summary.md"), args.run_tag)
    if args.checkpoint_comparison_out:
        args.checkpoint_comparison_out = with_run_tag(
            args.checkpoint_comparison_out,
            "",
            args.run_tag,
        )

    config = load_config(ROOT_DIR / Path(args.config))
    defaults = dict(config.get("defaults", {}))
    experiment_meta = resolve_experiment_meta(defaults)
    datasets = resolve_cli_subset(args.datasets, lambda item: canonicalize_dataset_name(str(item)))
    if datasets is None:
        datasets = [canonicalize_dataset_name(str(name)) for name in defaults.get("datasets", [])]
    horizons = resolve_cli_subset(args.horizons, int)
    if horizons is None:
        horizons = [int(item) for item in defaults.get("horizons", [])]
    seeds = resolve_cli_subset(args.seeds, int)
    if seeds is None:
        seeds = [int(item) for item in defaults.get("seeds", [0])]
    dataset_id_map = {name: idx for idx, name in enumerate(datasets)}
    horizon_id_map = {int(value): idx for idx, value in enumerate(horizons)}
    views_dir = ROOT_DIR / Path(args.views_dir)
    registry_path = ROOT_DIR / Path(args.registry)
    events_path = ROOT_DIR / Path(args.events)
    support_summary_path = ROOT_DIR / Path(args.support_summary)
    baseline_results_path = ROOT_DIR / Path(args.baseline_results)
    results_out = resolve_optional_output_path(args.results_out)
    results_online_out = resolve_optional_output_path(args.results_online_out)
    validity_out = resolve_optional_output_path(args.validity_out)
    window_errors_out = resolve_optional_output_path(args.window_errors_out)
    arg_out = resolve_optional_output_path(args.arg_out)
    wgr_out = resolve_optional_output_path(args.wgr_out)
    ri_out = resolve_optional_output_path(args.ri_out)
    report_out = resolve_optional_output_path(args.report_out)
    checkpoint_out = resolve_optional_output_path(args.checkpoint_comparison_out)
    results_out, results_online_out, validity_out = resolve_results_side_outputs(results_out, results_online_out, validity_out)
    val_debug_window_errors_out = make_val_debug_window_error_path(window_errors_out) if args.debug_write_val_window_errors else None
    output_paths = [
        path
        for path in [results_out, results_online_out, validity_out, arg_out, wgr_out, ri_out, report_out, checkpoint_out]
        if path is not None
    ]
    if args.write_window_errors and window_errors_out is not None:
        output_paths.append(window_errors_out)
    if val_debug_window_errors_out is not None:
        output_paths.append(val_debug_window_errors_out)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.write_window_errors and window_errors_out is not None and window_errors_out.exists():
        window_errors_out.unlink()
    if val_debug_window_errors_out is not None and val_debug_window_errors_out.exists():
        val_debug_window_errors_out.unlink()
    support_vocab = load_support_vocab(support_summary_path)

    result_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    log_progress(
        f"start run_tag={args.run_tag or 'default'} datasets={datasets} horizons={horizons} seeds={seeds} "
        f"write_window_errors={bool(args.write_window_errors)} rich_window_errors={bool(args.window_error_rich_fields)}"
    )

    for dataset_name in datasets:
        dataset_cfg = resolve_aif_plus_dataset_config(defaults, dataset_name, config_source_path=args.config)
        runtime_cfg = dict(dataset_cfg["runtime"])
        model_cfg = dict(dataset_cfg["model"])
        loss_cfg = dict(dataset_cfg["loss"])
        lookback = int(dataset_cfg["lookback"])
        collapse_aux_label_vocabs = bool(dataset_cfg["collapse_aux_label_vocabs"])
        dataset_bundle = load_dataset_bundle(dataset_name, registry_path=registry_path)
        events_lookup = load_events_lookup(events_path=events_path, dataset_name=dataset_name)
        clean_view_name = resolve_clean_view_name(config, dataset_name)

        for horizon in horizons:
            log_progress(f"prepare dataset={dataset_name} L{lookback} H{horizon}")
            view_df = load_view_frame(views_dir, dataset_name, lookback=lookback, horizon=horizon)
            train_rows = select_view_rows(
                view_df,
                split_name="train",
                view_name=clean_view_name,
                max_rows=runtime_cfg.get("max_train_windows"),
            )
            val_rows = select_validation_rows(
                view_df=view_df,
                clean_view_name=clean_view_name,
                max_rows=runtime_cfg.get("max_val_windows"),
                min_clean_val_windows=int(runtime_cfg.get("min_clean_val_windows", 1)),
            )
            if train_rows.empty or val_rows.empty:
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
            group_map = build_group_map(train_rows)
            metadata_context = {
                "dataset_id": dataset_id_map[dataset_name],
                "horizon_id": horizon_id_map[horizon],
                "support_train": support_id_lookup.get("train", 0),
                "support_val": support_id_lookup.get("val", 0),
                "support_test": support_id_lookup.get("test", 0),
            }

            for seed in seeds:
                log_prefix = f"{dataset_name}/L{lookback}/H{horizon}/seed{seed}"
                set_random_seed(seed)
                device = resolve_device(str(runtime_cfg.get("device", "auto")))
                model = AIFPlus(
                    build_model_config(
                        model_cfg=model_cfg,
                        lookback=lookback,
                        horizon=horizon,
                        n_vars=dataset_bundle.n_vars,
                        horizon_vocab_size=len(horizon_id_map),
                    )
                )
                model.to(device)
                ema = (
                    ModelEMA(
                        model,
                        decay=float(runtime_cfg.get("ema_decay", 0.999)),
                        device=str(runtime_cfg.get("ema_device", "same")),
                    )
                    if bool(runtime_cfg.get("use_ema", False))
                    else None
                )
                selected_checkpoint_variant = "best_ema_single" if ema is not None else "best_single"
                pin_memory = bool(runtime_cfg.get("pin_memory", True)) and device.type == "cuda"
                train_loader = build_loader(
                    AIFPlusWindowDataset(
                        rows=train_rows,
                        dataset_bundle=dataset_bundle,
                        events_lookup=events_lookup,
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

                train_info = fit_aif_plus_best_single(
                    model=model,
                    ema=ema,
                    train_loader=train_loader,
                    val_rows=val_rows,
                    dataset_bundle=dataset_bundle,
                    events_lookup=events_lookup,
                    clean_view_name=clean_view_name,
                    runtime_cfg=runtime_cfg,
                    loss_cfg=loss_cfg,
                    metadata_context=metadata_context,
                    group_map=group_map,
                    artifact_map=artifact_map,
                    phase_map=phase_map,
                    support_id_lookup=support_id_lookup,
                    horizon_id=horizon_id_map[horizon],
                    seed=seed,
                    log_prefix=log_prefix,
                    val_window_error_path=val_debug_window_errors_out,
                    window_error_rich_fields=bool(args.window_error_rich_fields),
                    val_setting_meta={
                        "dataset_name": dataset_name,
                        "backbone": experiment_meta["backbone_name"],
                        "lookback": lookback,
                        "horizon": horizon,
                        "train_view_name": experiment_meta["train_view_name"],
                        "eval_view_name": "val",
                        "seed": seed,
                        "checkpoint_variant": selected_checkpoint_variant,
                    },
                )

                load_model_state(model, train_info["best_state"])
                source_epochs_text = ",".join(str(item) for item in train_info["checkpoint_source_epochs"])
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
                    error_setting_meta = {
                        "dataset_name": dataset_name,
                        "backbone": experiment_meta["backbone_name"],
                        "lookback": lookback,
                        "horizon": horizon,
                        "train_view_name": experiment_meta["train_view_name"],
                        "eval_view_name": eval_view_name,
                        "seed": seed,
                        "checkpoint_variant": train_info["checkpoint_variant"],
                    }
                    metrics = evaluate_aif_plus(
                        model=model,
                        dataset_bundle=dataset_bundle,
                        events_lookup=events_lookup,
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
                        setting_meta=error_setting_meta,
                        window_error_path=window_errors_out,
                        write_window_errors=bool(args.write_window_errors) and window_errors_out is not None,
                        window_error_rich_fields=bool(args.window_error_rich_fields),
                    )
                    checkpoint_rows.append(
                        {
                            "dataset_name": dataset_name,
                            "backbone": experiment_meta["backbone_name"],
                            "lookback": lookback,
                            "horizon": horizon,
                            "train_view_name": experiment_meta["train_view_name"],
                            "eval_view_name": eval_view_name,
                            "seed": seed,
                            "checkpoint_variant": train_info["checkpoint_variant"],
                            "checkpoint_source_epochs": source_epochs_text,
                            "selected_for_final": 1,
                            "selection_status": train_info["selection_status"],
                            "hyperparam_source_kind": dataset_cfg["hyperparam_source_kind"],
                            "hyperparam_source_url": dataset_cfg["hyperparam_source_url"],
                            "hyperparam_source_note": dataset_cfg["hyperparam_source_note"],
                            "n_train_windows": int(len(train_rows)),
                            "n_eval_windows": int(metrics["n_total_windows"]),
                            "n_total_windows": int(metrics["n_total_windows"]),
                            "n_valid_windows": int(metrics["n_valid_windows"]),
                            "n_invalid_windows": int(metrics["n_invalid_windows"]),
                            "invalid_ratio": round(float(metrics["invalid_ratio"]), 6),
                            "epochs_ran": int(train_info["epochs_ran"]),
                            "best_epoch": int(train_info["best_epoch"]),
                            "val_mae": round(float(train_info["best_val_metrics"]["mae"]), 6),
                            "val_mse": round(float(train_info["best_val_metrics"]["mse"]), 6),
                            "val_score": round(float(train_info["best_val_score"]), 6),
                            "mae": round(float(metrics["mae"]), 6),
                            "mse": round(float(metrics["mse"]), 6),
                            "smape": round(float(metrics["smape"]), 6),
                        }
                    )
                    result_rows.append(
                        {
                            "dataset_name": dataset_name,
                            "backbone": experiment_meta["backbone_name"],
                            "lookback": lookback,
                            "horizon": horizon,
                            "train_view_name": experiment_meta["train_view_name"],
                            "eval_view_name": eval_view_name,
                            "seed": seed,
                            "checkpoint_variant": train_info["checkpoint_variant"],
                            "checkpoint_source_epochs": source_epochs_text,
                            "hyperparam_source_kind": dataset_cfg["hyperparam_source_kind"],
                            "hyperparam_source_url": dataset_cfg["hyperparam_source_url"],
                            "hyperparam_source_note": dataset_cfg["hyperparam_source_note"],
                            "n_train_windows": int(len(train_rows)),
                            "n_eval_windows": int(metrics["n_total_windows"]),
                            "n_total_windows": int(metrics["n_total_windows"]),
                            "n_valid_windows": int(metrics["n_valid_windows"]),
                            "n_invalid_windows": int(metrics["n_invalid_windows"]),
                            "invalid_ratio": round(float(metrics["invalid_ratio"]), 6),
                            "epochs_ran": int(train_info["epochs_ran"]),
                            "best_epoch": int(train_info["best_epoch"]),
                            "best_val_mae": round(float(train_info["best_val_metrics"]["mae"]), 6),
                            "best_val_mse": round(float(train_info["best_val_metrics"]["mse"]), 6),
                            "best_val_score": round(float(train_info["best_val_score"]), 6),
                            "mae": round(float(metrics["mae"]), 6),
                            "mse": round(float(metrics["mse"]), 6),
                            "smape": round(float(metrics["smape"]), 6),
                        }
                    )
                    log_progress(
                        f"done dataset={dataset_name} L{lookback} H{horizon} seed={seed} "
                        f"variant={train_info['checkpoint_variant']} view={eval_view_name} "
                        f"mae={float(metrics['mae']):.6f} valid={int(metrics['n_valid_windows'])}/{int(metrics['n_total_windows'])}"
                    )
                del train_info
                del train_loader
                del ema
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del events_lookup
        del dataset_bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    online_results_df = sort_results_frame(pd.DataFrame(result_rows))
    checkpoint_df = sort_results_frame(pd.DataFrame(checkpoint_rows))
    window_errors_df = (
        pd.read_csv(window_errors_out, low_memory=False)
        if args.write_window_errors and window_errors_out is not None and window_errors_out.exists()
        else pd.DataFrame()
    )
    results_df, validity_df = recompute_results_from_window_errors(online_results_df, window_errors_df)
    arg_df = compute_aif_arg_table(results_df) if arg_out is not None or report_out is not None else pd.DataFrame()
    ri_df = compute_aif_ri_table(results_df) if ri_out is not None or report_out is not None else pd.DataFrame()
    wgr_input = window_errors_df.copy()
    if not wgr_input.empty and "is_valid_metric" in wgr_input.columns:
        wgr_input = wgr_input[wgr_input["is_valid_metric"].fillna(0).astype(int) == 1].copy()
    wgr_df = compute_aif_wgr_table(wgr_input) if wgr_out is not None or report_out is not None else pd.DataFrame()

    if results_online_out is not None:
        online_results_df.to_csv(results_online_out, index=False)
    if results_out is not None:
        results_df.to_csv(results_out, index=False)
    if validity_out is not None:
        validity_df.to_csv(validity_out, index=False)
    if checkpoint_out is not None:
        checkpoint_df.to_csv(checkpoint_out, index=False)
    if arg_out is not None:
        arg_df.to_csv(arg_out, index=False)
    if wgr_out is not None:
        wgr_df.to_csv(wgr_out, index=False)
    if ri_out is not None:
        ri_df.to_csv(ri_out, index=False)

    if report_out is not None:
        baseline_df = pd.read_csv(baseline_results_path, low_memory=False) if baseline_results_path.exists() else pd.DataFrame()
        write_markdown(
            report_out,
            build_summary_markdown(
                results_df,
                baseline_df,
                arg_df,
                wgr_df,
                ri_df,
                checkpoint_df=checkpoint_df,
                summary_title=experiment_meta["summary_title"],
                display_name=experiment_meta["display_name"],
                summary_note=experiment_meta["summary_note"],
            ),
        )
    log_progress(
        f"finished results_out={results_out} results_online_out={results_online_out} rows={len(results_df)}"
    )


if __name__ == "__main__":
    main()
