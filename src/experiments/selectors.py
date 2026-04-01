from __future__ import annotations

from dataclasses import dataclass, field

from .spec import ExperimentSpec


def _normalized(values: set[str]) -> set[str]:
    return {item.strip() for item in values if item.strip()}


@dataclass(frozen=True)
class RunSelector:
    setting_ids: set[str] = field(default_factory=set)
    datasets: set[str] = field(default_factory=set)
    backbones: set[str] = field(default_factory=set)
    horizons: set[int] = field(default_factory=set)
    seeds: set[int] = field(default_factory=set)
    train_views: set[str] = field(default_factory=set)
    eval_views: set[str] = field(default_factory=set)
    start_setting_index: int | None = None
    end_setting_index: int | None = None
    shard_id: int | None = None
    num_shards: int | None = None

    def validate(self) -> None:
        if self.shard_id is None and self.num_shards is None:
            return
        if self.shard_id is None or self.num_shards is None:
            raise ValueError("Both shard_id and num_shards must be set together.")
        if self.num_shards <= 0:
            raise ValueError("num_shards must be positive.")
        if self.shard_id < 0 or self.shard_id >= self.num_shards:
            raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards.")


def select_specs(specs: list[ExperimentSpec], selector: RunSelector) -> list[ExperimentSpec]:
    selector.validate()
    selected: list[ExperimentSpec] = []
    setting_ids = _normalized(selector.setting_ids)
    datasets = _normalized(selector.datasets)
    backbones = _normalized(selector.backbones)
    train_views = _normalized(selector.train_views)
    eval_views = _normalized(selector.eval_views)

    for spec in sorted(specs, key=lambda item: item.setting_index):
        if setting_ids and spec.setting_id not in setting_ids:
            continue
        if datasets and spec.dataset_name not in datasets:
            continue
        if backbones and spec.backbone not in backbones:
            continue
        if selector.horizons and spec.horizon not in selector.horizons:
            continue
        if selector.seeds and spec.seed not in selector.seeds:
            continue
        if train_views and spec.train_view_name not in train_views:
            continue
        if eval_views and spec.eval_view_name not in eval_views:
            continue
        if selector.start_setting_index is not None and spec.setting_index < selector.start_setting_index:
            continue
        if selector.end_setting_index is not None and spec.setting_index > selector.end_setting_index:
            continue
        selected.append(spec)

    if selector.shard_id is None or selector.num_shards is None:
        return selected
    return [spec for position, spec in enumerate(selected) if position % selector.num_shards == selector.shard_id]
