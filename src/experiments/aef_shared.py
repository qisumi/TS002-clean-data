from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training import build_grad_scaler, resolve_device, set_random_seed
from training.logging import log_progress as _default_log_progress


class TabularFeatureEncoder:
    def __init__(self, categorical_cols: list[str]) -> None:
        self.categorical_cols = categorical_cols
        self.numeric_cols: list[str] = []
        self.levels: dict[str, list[str]] = {}
        self.feature_names: list[str] = []

    def fit(self, frame: pd.DataFrame) -> None:
        excluded = {"window_id", *self.categorical_cols}
        self.numeric_cols = [col for col in frame.columns if col not in excluded]
        self.levels = {
            col: sorted(frame[col].fillna("NA").astype(str).unique().tolist())
            for col in self.categorical_cols
        }
        self.feature_names = list(self.numeric_cols)
        for col in self.categorical_cols:
            self.feature_names.extend([f"{col}={level}" for level in self.levels[col]])

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        pieces: list[np.ndarray] = []
        if self.numeric_cols:
            pieces.append(frame.loc[:, self.numeric_cols].fillna(0.0).to_numpy(dtype=np.float32))

        for col in self.categorical_cols:
            levels = self.levels[col]
            mapping = {level: idx for idx, level in enumerate(levels)}
            values = frame[col].fillna("NA").astype(str).to_numpy()
            encoded = np.zeros((len(frame), len(levels)), dtype=np.float32)
            for row_idx, value in enumerate(values):
                col_idx = mapping.get(value)
                if col_idx is not None:
                    encoded[row_idx, col_idx] = 1.0
            pieces.append(encoded)

        if not pieces:
            return np.zeros((len(frame), 0), dtype=np.float32)
        return np.concatenate(pieces, axis=1)


class TabularForecastDataset(Dataset[dict[str, Any]]):
    def __init__(self, features: np.ndarray, targets: np.ndarray, metadata: pd.DataFrame) -> None:
        self.features = features.astype(np.float32, copy=False)
        self.targets = targets.astype(np.float32, copy=False)
        self.metadata = metadata.reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]
        return {
            "x": torch.from_numpy(self.features[index]),
            "y": torch.from_numpy(self.targets[index]),
            "window_id": str(row["window_id"]),
            "group_key": str(row.get("primary_group_key", "NA")),
            "phase_group": str(row.get("dominant_phase_target", "NA")),
            "artifact_group_major": str(row.get("artifact_group_major", "NA")),
            "is_flagged": int(row.get("is_flagged", 0)),
        }


class AEFRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def longest_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for item in mask.astype(bool).tolist():
        if item:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def extract_feature_record(row: pd.Series, scaled_window: np.ndarray, raw_window: np.ndarray, feature_cfg: dict[str, Any]) -> dict[str, Any]:
    diffs = np.diff(scaled_window, axis=0) if len(scaled_window) > 1 else np.zeros((0, scaled_window.shape[1]), dtype=np.float32)
    last_step = scaled_window[-1]
    row_zero_ratio = np.abs(raw_window) <= float(feature_cfg.get("near_zero_tolerance", 1e-8))
    row_zero_share = row_zero_ratio.mean(axis=1)
    row_diff = np.abs(np.diff(raw_window, axis=0)).mean(axis=1) if len(raw_window) > 1 else np.zeros(0, dtype=np.float32)

    return {
        "window_id": str(row["window_id"]),
        "art_input_contam_score": float(row["input_contam_score"]),
        "art_target_contam_score": float(row["target_contam_score"]),
        "art_n_events_input": float(row["n_events_input"]),
        "art_n_events_target": float(row["n_events_target"]),
        "art_max_event_weight_input": float(row["max_event_weight_input"]),
        "art_max_event_weight_target": float(row["max_event_weight_target"]),
        "art_has_corrupted_target": float(row["has_corrupted_target"]),
        "art_has_suspicious_target": float(row["has_suspicious_target"]),
        "art_has_valid_high_target": float(row["has_valid_high_target"]),
        "art_repairable_input_overlap": float(row["repairable_input_overlap"]),
        "art_unrecoverable_input_overlap": float(row["unrecoverable_input_overlap"]),
        "art_has_unrecoverable_target": float(row["has_unrecoverable_target"]),
        "art_has_active_suspicious_target": float(row["has_active_suspicious_target"]),
        "art_is_flagged": float(row["is_flagged"]),
        "phase_input_active": float(row["phase_share_input_active"]),
        "phase_input_transition": float(row["phase_share_input_transition"]),
        "phase_input_night": float(row["phase_share_input_night"]),
        "phase_target_active": float(row["phase_share_target_active"]),
        "phase_target_transition": float(row["phase_share_target_transition"]),
        "phase_target_night": float(row["phase_share_target_night"]),
        "stat_last_value_mean": float(last_step.mean()),
        "stat_last_value_std": float(last_step.std()),
        "stat_last_value_min": float(last_step.min()),
        "stat_last_value_max": float(last_step.max()),
        "stat_input_mean": float(scaled_window.mean()),
        "stat_input_std": float(scaled_window.std()),
        "stat_input_var_mean": float(np.var(scaled_window, axis=0).mean()),
        "stat_last_diff_mean": float(diffs.mean()) if diffs.size else 0.0,
        "stat_last_diff_std": float(diffs.std()) if diffs.size else 0.0,
        "stat_row_zero_ratio": float(row_zero_ratio.mean()),
        "stat_flat_run_max": float(longest_run(row_diff <= float(feature_cfg.get("flat_tolerance", 1e-8))) + 1),
        "stat_zero_block_max": float(longest_run(row_zero_share >= 0.5)),
        "cat_dataset_name": str(row.get("dataset_name", "NA")),
        "cat_horizon": str(row.get("horizon", "NA")),
        "cat_artifact_group_major": str(row.get("artifact_group_major", "NA")),
        "cat_dominant_phase_input": str(row.get("dominant_phase_input", "NA")),
        "cat_dominant_phase_target": str(row.get("dominant_phase_target", "NA")),
        "cat_severity_bin": str(row.get("severity_bin", "NA")),
        "cat_n_variables_bin": str(row.get("n_variables_bin", "NA")),
        "cat_phase_mix_bin_target": str(row.get("phase_mix_bin_target", "NA")),
    }


def build_feature_frame(rows: pd.DataFrame, bundle: Any, feature_cfg: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        input_start = int(row.input_start)
        input_end = int(row.input_end)
        scaled_window = bundle.scaled_values[input_start : input_end + 1]
        raw_window = bundle.raw_values[input_start : input_end + 1]
        records.append(
            extract_feature_record(pd.Series(row._asdict()), scaled_window=scaled_window, raw_window=raw_window, feature_cfg=feature_cfg)
        )
    return pd.DataFrame(records)


def build_targets(rows: pd.DataFrame, bundle: Any) -> np.ndarray:
    targets: list[np.ndarray] = []
    for row in rows.itertuples(index=False):
        target = bundle.scaled_values[int(row.target_start) : int(row.target_end) + 1]
        targets.append(target.reshape(-1).astype(np.float32, copy=False))
    return np.stack(targets, axis=0) if targets else np.zeros((0, 0), dtype=np.float32)


def control_feature_mask(feature_names: list[str]) -> np.ndarray:
    tokens = ("art_", "phase_", "cat_artifact_", "cat_dominant_phase_", "cat_severity_", "cat_n_variables_", "cat_phase_mix_")
    return np.array([any(name.startswith(token) for token in tokens) for name in feature_names], dtype=bool)


def shuffle_control_features(features: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
    if features.size == 0 or not mask.any():
        return features
    rng = np.random.default_rng(seed)
    out = features.copy()
    for col_idx in np.flatnonzero(mask):
        permutation = rng.permutation(out.shape[0])
        out[:, col_idx] = out[permutation, col_idx]
    return out


def build_loader(dataset: Dataset[Any], batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def blocked_permute_frame(
    frame: pd.DataFrame,
    cols_to_shuffle: list[str],
    block_cols: list[str],
    seed: int,
) -> pd.DataFrame:
    out = frame.copy()
    if not cols_to_shuffle:
        return out
    rng = np.random.default_rng(seed)
    valid_block_cols = [col for col in block_cols if col in out.columns]
    if not valid_block_cols:
        valid_block_cols = [out.index.name] if out.index.name else []
    if not valid_block_cols:
        permutation = rng.permutation(len(out))
        out.loc[:, cols_to_shuffle] = out.iloc[permutation][cols_to_shuffle].to_numpy()
        return out

    for _, idx in out.groupby(valid_block_cols, dropna=False).groups.items():
        idx_list = list(idx)
        if len(idx_list) <= 1:
            continue
        perm = rng.permutation(idx_list)
        for col in cols_to_shuffle:
            out.loc[idx_list, col] = out.loc[perm, col].to_numpy()
    return out


def artifact_phase_cols(frame: pd.DataFrame) -> list[str]:
    prefixes = (
        "art_",
        "phase_",
        "cat_artifact_",
        "cat_dominant_phase_",
        "cat_severity_",
        "cat_n_variables_",
        "cat_phase_mix_",
    )
    return [col for col in frame.columns if any(col.startswith(prefix) for prefix in prefixes)]


def series_only_cols(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col.startswith("stat_")]


def fit_aef_variant(
    model_name: str,
    train_features_df: pd.DataFrame,
    val_features_df: pd.DataFrame,
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    runtime_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    seed: int,
    horizon: int,
    dataset_name: str,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[nn.Module, TabularFeatureEncoder, float, int]:
    block_cols = ["cat_dataset_name", "cat_horizon", "cat_dominant_phase_target", "cat_severity_bin"]
    shuffle_cols = artifact_phase_cols(train_features_df)

    if model_name == "AEF-Strong":
        variant_train_df = train_features_df.copy()
        variant_val_df = val_features_df.copy()
    elif model_name == "AEF-ControlBlocked":
        variant_train_df = blocked_permute_frame(train_features_df, shuffle_cols, block_cols, seed=seed + 11)
        variant_val_df = blocked_permute_frame(val_features_df, shuffle_cols, block_cols, seed=seed + 29)
    elif model_name == "AEF-SeriesOnly":
        keep_cols = series_only_cols(train_features_df)
        variant_train_df = train_features_df.loc[:, keep_cols].copy()
        variant_val_df = val_features_df.loc[:, keep_cols].copy()
    else:
        raise ValueError(f"Unsupported model_name={model_name}")

    categorical_cols = [col for col in variant_train_df.columns if col.startswith("cat_")]
    encoder = TabularFeatureEncoder(categorical_cols=categorical_cols)
    encoder.fit(variant_train_df)
    train_x = encoder.transform(variant_train_df)
    val_x = encoder.transform(variant_val_df)

    model = AEFRegressor(
        input_dim=int(train_x.shape[1]),
        output_dim=int(train_targets.shape[1]),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        hidden_layers=int(model_cfg.get("hidden_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    model, best_val_mae, epochs_ran = fit_aef_model(
        model=model,
        train_dataset=TabularForecastDataset(train_x, train_targets, train_rows),
        val_dataset=TabularForecastDataset(val_x, val_targets, val_rows),
        runtime_cfg=runtime_cfg,
        seed=seed,
        log_prefix=f"{dataset_name}/H{horizon}/{model_name}",
        log_fn=log_fn,
    )
    return model, encoder, float(best_val_mae), int(epochs_ran)


def fit_aef_model(
    model: nn.Module,
    train_dataset: TabularForecastDataset,
    val_dataset: TabularForecastDataset,
    runtime_cfg: dict[str, Any],
    seed: int,
    log_prefix: str = "",
    log_fn: Callable[[str], None] | None = None,
) -> tuple[nn.Module, float, int]:
    logger = log_fn or _default_log_progress
    set_random_seed(seed)
    device = resolve_device(str(runtime_cfg.get("device", "auto")))
    pin_memory = bool(runtime_cfg.get("pin_memory", True)) and device.type == "cuda"
    amp_enabled = bool(runtime_cfg.get("amp", True)) and device.type == "cuda"
    scaler = build_grad_scaler(device, enabled=amp_enabled)

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(runtime_cfg.get("lr", 1e-3)),
        weight_decay=float(runtime_cfg.get("weight_decay", 0.0)),
    )
    train_loader = build_loader(
        train_dataset,
        batch_size=int(runtime_cfg.get("batch_size", 128)),
        shuffle=True,
        num_workers=int(runtime_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=int(runtime_cfg.get("eval_batch_size", 256)),
        shuffle=False,
        num_workers=int(runtime_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )

    best_state = None
    best_val_mae = float("inf")
    patience = int(runtime_cfg.get("patience", 3))
    patience_counter = 0
    epochs_ran = 0
    prefix = f"{log_prefix} " if log_prefix else ""
    logger(f"{prefix}fit start device={device.type} train_rows={len(train_dataset)} val_rows={len(val_dataset)}")

    for epoch in range(1, int(runtime_cfg.get("epochs", 10)) + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(x)
                loss = F.mse_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        val_mae = evaluate_aef_model(model, val_loader, device=device, amp_enabled=amp_enabled, meta=None)[0]["mae"]
        epochs_ran = epoch
        logger(
            f"{prefix}epoch {epoch}/{int(runtime_cfg.get('epochs', 10))} "
            f"val_mae={float(val_mae):.6f} best_val_mae={min(best_val_mae, float(val_mae)):.6f}"
        )
        if val_mae + 1e-6 < best_val_mae:
            best_val_mae = float(val_mae)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger(f"{prefix}fit done epochs_ran={epochs_ran} best_val_mae={best_val_mae:.6f}")
    return model, float(best_val_mae), epochs_ran


@torch.no_grad()
def evaluate_aef_model(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    amp_enabled: bool,
    meta: dict[str, Any] | None,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    total_smape = 0.0
    total_count = 0
    rows: list[dict[str, Any]] = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            pred = model(x)

        mae_vec = (pred - y).abs().mean(dim=1)
        mse_vec = ((pred - y) ** 2).mean(dim=1)
        smape_vec = (200.0 * (pred - y).abs() / (pred.abs() + y.abs() + 1e-6)).mean(dim=1)

        batch_size = int(x.shape[0])
        total_mae += float(mae_vec.sum().item())
        total_mse += float(mse_vec.sum().item())
        total_smape += float(smape_vec.sum().item())
        total_count += batch_size

        for idx in range(batch_size):
            rows.append(
                {
                    **(meta or {}),
                    "window_id": str(batch["window_id"][idx]),
                    "group_key": str(batch["group_key"][idx]),
                    "phase_group": str(batch["phase_group"][idx]),
                    "artifact_group_major": str(batch["artifact_group_major"][idx]),
                    "is_flagged": int(batch["is_flagged"][idx]),
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


def compare_with_standard(results_df: pd.DataFrame, baseline_results: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline_results[
        (baseline_results["status"] == "completed")
        & (baseline_results["train_view_name"] == "raw")
    ].copy()
    best_standard = (
        baseline.groupby(["dataset_name", "horizon", "eval_view_name"], dropna=False)["mae"]
        .min()
        .reset_index()
        .rename(columns={"mae": "best_forecaster_mae"})
    )
    return results_df.merge(best_standard, on=["dataset_name", "horizon", "eval_view_name"], how="left")


__all__ = [
    "AEFRegressor",
    "TabularFeatureEncoder",
    "TabularForecastDataset",
    "artifact_phase_cols",
    "blocked_permute_frame",
    "build_feature_frame",
    "build_loader",
    "build_targets",
    "compare_with_standard",
    "control_feature_mask",
    "evaluate_aef_model",
    "fit_aef_variant",
    "fit_aef_model",
    "load_config",
    "series_only_cols",
    "shuffle_control_features",
]
