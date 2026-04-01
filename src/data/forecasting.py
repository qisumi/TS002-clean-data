from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.dataset_bundle import read_dataset
from data.paths import ROOT_DIR
from views.selection import resolve_dataset_splits

DEFAULT_CYCLE_LENGTHS = {
    "ETTh1": 24,
    "ETTh2": 24,
    "ETTm1": 96,
    "ETTm2": 96,
    "solar_AL": 144,
    "weather": 144,
    "exchange_rate": 7,
    "electricity": 168,
}


@dataclass
class DatasetBundle:
    dataset_name: str
    raw_values: np.ndarray
    scaled_values: np.ndarray
    column_names: list[str]
    column_index: dict[str, int]
    train_mean: np.ndarray
    train_std: np.ndarray
    cycle_len: int

    @property
    def n_vars(self) -> int:
        return int(self.scaled_values.shape[1])


def resolve_cycle_length(dataset_name: str, params: dict[str, Any]) -> int:
    if "cycle" in params:
        return int(params["cycle"])
    return int(DEFAULT_CYCLE_LENGTHS.get(dataset_name, 24))


def load_dataset_bundle(dataset_name: str, registry_path: Path) -> DatasetBundle:
    registry = pd.read_csv(registry_path)
    registry_row = registry.loc[registry["dataset_name"] == dataset_name].iloc[0]
    file_path = Path(str(registry_row["file_path"]))
    if not file_path.is_absolute():
        file_path = ROOT_DIR / file_path

    bundle = read_dataset(file_path)
    raw_values = bundle.frame.loc[:, bundle.numeric_columns].to_numpy(dtype=np.float32)
    column_names = [str(item) for item in bundle.numeric_columns]
    column_index = {name: idx for idx, name in enumerate(column_names)}

    split_start, split_end = resolve_dataset_splits(dataset_name, len(raw_values))["train"]
    train_values = raw_values[split_start : split_end + 1]
    train_mean = np.mean(train_values, axis=0, dtype=np.float64).astype(np.float32)
    train_std = np.std(train_values, axis=0, dtype=np.float64).astype(np.float32)
    train_std = np.where(train_std < 1e-6, 1.0, train_std).astype(np.float32)
    scaled_values = ((raw_values - train_mean) / train_std).astype(np.float32)

    return DatasetBundle(
        dataset_name=dataset_name,
        raw_values=raw_values,
        scaled_values=scaled_values,
        column_names=column_names,
        column_index=column_index,
        train_mean=train_mean,
        train_std=train_std,
        cycle_len=resolve_cycle_length(dataset_name, {}),
    )


def load_view_frame(views_dir: Path, dataset_name: str, lookback: int, horizon: int) -> pd.DataFrame:
    path = views_dir / f"{dataset_name}_L{lookback}_H{horizon}.csv"
    return pd.read_csv(path, low_memory=False)


def load_events_lookup(events_path: Path, dataset_name: str) -> dict[str, dict[str, Any]]:
    events = pd.read_csv(events_path, low_memory=False)
    subset = events.loc[events["dataset_name"] == dataset_name].copy()
    return {str(row["artifact_id"]): row.to_dict() for _, row in subset.iterrows()}
