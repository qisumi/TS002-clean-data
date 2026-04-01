from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

from backbones.factory import forward_backbone
from data.forecasting import DatasetBundle
from training.dataloaders import build_dataloader
from training.logging import log_progress


@torch.no_grad()
def evaluate_forecaster(
    model: nn.Module,
    dataset_bundle: DatasetBundle,
    events_lookup: dict[str, dict[str, Any]],
    eval_rows: pd.DataFrame,
    runtime_cfg: dict[str, Any],
    apply_intervention: bool,
    setting_meta: dict[str, Any] | None,
    collect_error_rows: bool = True,
    log_path: str | Path | None = None,
    log_prefix: str = "",
) -> tuple[dict[str, float], pd.DataFrame]:
    device = next(model.parameters()).device
    amp_enabled = bool(runtime_cfg.get("amp", True)) and device.type == "cuda"
    pin_memory = bool(runtime_cfg.get("pin_memory", True)) and device.type == "cuda"
    eval_batch_size = int(runtime_cfg.get("eval_batch_size", 128))
    prefix = f"{log_prefix} " if log_prefix else ""

    while True:
        loader = build_dataloader(
            rows=eval_rows,
            dataset_bundle=dataset_bundle,
            events_lookup=events_lookup,
            apply_intervention=apply_intervention,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=int(runtime_cfg.get("num_workers", 0)),
            pin_memory=pin_memory,
        )
        try:
            model.eval()
            total_mae = 0.0
            total_mse = 0.0
            total_smape = 0.0
            total_count = 0
            error_rows: list[dict[str, Any]] = []

            for batch in loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                cycle_index = batch["cycle_index"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    pred = forward_backbone(model, x, cycle_index=cycle_index)

                mae_vec = (pred - y).abs().mean(dim=(1, 2))
                mse_vec = ((pred - y) ** 2).mean(dim=(1, 2))
                smape_vec = (200.0 * (pred - y).abs() / (pred.abs() + y.abs() + 1e-6)).mean(dim=(1, 2))

                batch_size = int(x.shape[0])
                total_mae += float(mae_vec.sum().item())
                total_mse += float(mse_vec.sum().item())
                total_smape += float(smape_vec.sum().item())
                total_count += batch_size

                if collect_error_rows:
                    meta = setting_meta or {}
                    for idx in range(batch_size):
                        error_rows.append(
                            {
                                **meta,
                                "window_id": str(batch["window_id"][idx]),
                                "group_key": str(batch["primary_group_key"][idx]),
                                "phase_group": str(batch["phase_group"][idx]),
                                "artifact_group_major": str(batch["artifact_group_major"][idx]),
                                "is_flagged": int(batch["is_flagged"][idx]),
                                "has_input_intervention": int(batch["has_input_intervention"][idx]),
                                "strict_target_clean": int(batch["strict_target_clean"][idx]),
                                "subset_name": str(batch["subset_name"][idx]),
                                "mae": float(mae_vec[idx].detach().cpu().item()),
                                "mse": float(mse_vec[idx].detach().cpu().item()),
                                "smape": float(smape_vec[idx].detach().cpu().item()),
                            }
                        )

                del x, y, cycle_index, pred, mae_vec, mse_vec, smape_vec

            if total_count == 0:
                return {"mae": float("nan"), "mse": float("nan"), "smape": float("nan")}, pd.DataFrame(error_rows)

            metrics = {
                "mae": float(total_mae / total_count),
                "mse": float(total_mse / total_count),
                "smape": float(total_smape / total_count),
            }
            return metrics, pd.DataFrame(error_rows)
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda" or eval_batch_size <= 1:
                raise
            next_batch_size = max(1, eval_batch_size // 2)
            log_progress(
                f"{prefix}evaluate_forecaster OOM at eval_batch_size={eval_batch_size}; retry with eval_batch_size={next_batch_size}",
                log_path=log_path,
            )
            eval_batch_size = next_batch_size
            del loader
            gc.collect()
            torch.cuda.empty_cache()
