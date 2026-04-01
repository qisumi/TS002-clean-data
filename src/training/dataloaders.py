from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.forecasting import DatasetBundle
from views.intervention import apply_intervention_recipe


class ForecastWindowDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: pd.DataFrame,
        dataset_bundle: DatasetBundle,
        events_lookup: dict[str, dict[str, Any]],
        apply_intervention: bool,
    ) -> None:
        ordered = rows.sort_values("target_start").reset_index(drop=True).copy()
        self.samples = ordered.to_dict(orient="records")
        self.dataset_bundle = dataset_bundle
        self.events_lookup = events_lookup
        self.apply_intervention = apply_intervention

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        input_start = int(sample["input_start"])
        input_end = int(sample["input_end"])
        target_start = int(sample["target_start"])
        target_end = int(sample["target_end"])

        x = self.dataset_bundle.scaled_values[input_start : input_end + 1].copy()
        y = self.dataset_bundle.scaled_values[target_start : target_end + 1].copy()
        cycle_len = int(getattr(self.dataset_bundle, "cycle_len", 1))
        cycle_index = int(input_start % max(cycle_len, 1))

        if self.apply_intervention:
            x = apply_intervention_recipe(
                input_window=x,
                global_input_start=input_start,
                recipe_text=str(sample.get("intervention_recipe", "")),
                events_lookup=self.events_lookup,
                column_index=self.dataset_bundle.column_index,
            )

        return {
            "x": torch.from_numpy(x.astype(np.float32, copy=False)),
            "y": torch.from_numpy(y.astype(np.float32, copy=False)),
            "window_id": str(sample["window_id"]),
            "primary_group_key": str(sample.get("primary_group_key", "NA")),
            "phase_group": str(sample.get("dominant_phase_target", "NA")),
            "is_flagged": int(sample.get("is_flagged", 0)),
            "artifact_group_major": str(sample.get("artifact_group_major", "NA")),
            "has_input_intervention": int(sample.get("has_input_intervention", 0)),
            "strict_target_clean": int(sample.get("strict_target_clean", 0)),
            "subset_name": str(sample.get("subset_name", "")),
            "cycle_index": torch.tensor(cycle_index, dtype=torch.long),
        }


def build_dataloader(
    rows: pd.DataFrame,
    dataset_bundle: DatasetBundle,
    events_lookup: dict[str, dict[str, Any]],
    apply_intervention: bool,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[Any]:
    dataset = ForecastWindowDataset(
        rows=rows,
        dataset_bundle=dataset_bundle,
        events_lookup=events_lookup,
        apply_intervention=apply_intervention,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
