from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from backbones.factory import forward_backbone, instantiate_backbone
from data.forecasting import DatasetBundle
from training.dataloaders import ForecastWindowDataset, build_dataloader
from training.evaluators import evaluate_forecaster
from training.logging import log_progress
from training.runtime import build_grad_scaler, resolve_device, set_random_seed


@dataclass
class TrainArtifacts:
    model: nn.Module
    train_view_name: str
    val_view_name: str
    best_val_mae: float
    epochs_ran: int
    device: str
    fit_seconds: float


def fit_forecaster(
    backbone_name: str,
    model_params: dict[str, Any],
    runtime_cfg: dict[str, Any],
    dataset_bundle: DatasetBundle,
    events_lookup: dict[str, dict[str, Any]],
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    train_view_name: str,
    val_view_name: str,
    seed: int,
    log_prefix: str = "",
    log_path: str | Path | None = None,
) -> TrainArtifacts:
    set_random_seed(seed)

    seq_len = int(train_rows["lookback"].iloc[0])
    pred_len = int(train_rows["horizon"].iloc[0])
    device = resolve_device(str(runtime_cfg.get("device", "auto")))
    amp_enabled = bool(runtime_cfg.get("amp", True)) and device.type == "cuda"
    scaler = build_grad_scaler(device, enabled=amp_enabled)
    pin_memory = bool(runtime_cfg.get("pin_memory", True)) and device.type == "cuda"
    model = instantiate_backbone(
        backbone_name,
        seq_len=seq_len,
        pred_len=pred_len,
        n_vars=dataset_bundle.n_vars,
        params=model_params,
        dataset_name=dataset_bundle.dataset_name,
    )
    model.to(device)
    prefix = f"{log_prefix} " if log_prefix else ""
    log_progress(
        f"{prefix}fit_forecaster start backbone={backbone_name} device={device.type} "
        f"train_windows={len(train_rows)} val_windows={len(val_rows)}",
        log_path=log_path,
    )

    train_loader = build_dataloader(
        rows=train_rows,
        dataset_bundle=dataset_bundle,
        events_lookup=events_lookup,
        apply_intervention=(train_view_name == "intervened"),
        batch_size=int(runtime_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(runtime_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )
    _ = build_dataloader(
        rows=val_rows,
        dataset_bundle=dataset_bundle,
        events_lookup=events_lookup,
        apply_intervention=(val_view_name == "intervened"),
        batch_size=int(runtime_cfg.get("eval_batch_size", 128)),
        shuffle=False,
        num_workers=int(runtime_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(runtime_cfg.get("lr", 1e-3)),
        weight_decay=float(runtime_cfg.get("weight_decay", 0.0)),
    )
    grad_clip = float(runtime_cfg.get("grad_clip", 0.0))
    epochs = int(runtime_cfg.get("epochs", 8))
    patience = int(runtime_cfg.get("patience", 3))

    best_state: dict[str, Any] | None = None
    best_val_mae = float("inf")
    patience_counter = 0
    epochs_ran = 0
    fit_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_count = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            cycle_index = batch["cycle_index"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                pred = forward_backbone(model, x, cycle_index=cycle_index)
                loss = F.mse_loss(pred, y)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_count += int(x.shape[0])

        val_metrics, _ = evaluate_forecaster(
            model=model,
            dataset_bundle=dataset_bundle,
            events_lookup=events_lookup,
            eval_rows=val_rows,
            runtime_cfg=runtime_cfg,
            apply_intervention=(val_view_name == "intervened"),
            setting_meta=None,
            collect_error_rows=False,
            log_path=log_path,
            log_prefix=f"{prefix}val",
        )
        val_mae = float(val_metrics["mae"])
        epochs_ran = epoch
        log_progress(
            f"{prefix}epoch {epoch}/{epochs} val_mae={val_mae:.6f} best_val_mae={min(best_val_mae, val_mae):.6f}",
            log_path=log_path,
        )

        if val_mae + 1e-6 < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if train_count == 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    fit_seconds = time.perf_counter() - fit_start
    log_progress(
        f"{prefix}fit_forecaster done backbone={backbone_name} epochs_ran={epochs_ran} "
        f"best_val_mae={best_val_mae:.6f} fit_seconds={fit_seconds:.2f}",
        log_path=log_path,
    )
    return TrainArtifacts(
        model=model,
        train_view_name=train_view_name,
        val_view_name=val_view_name,
        best_val_mae=float(best_val_mae),
        epochs_ran=epochs_ran,
        device=str(device),
        fit_seconds=float(round(fit_seconds, 3)),
    )
