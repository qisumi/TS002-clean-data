from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentSpec:
    task_name: str
    setting_id: str
    setting_index: int
    dataset_name: str
    backbone: str
    lookback: int
    horizon: int
    seed: int
    train_view_token: str
    eval_view_token: str
    train_view_name: str
    eval_view_name: str
    eval_row_view_name: str
    eval_protocol: str
    subset_name: str
    apply_eval_intervention: bool
    runtime_cfg: dict[str, Any] = field(default_factory=dict)
    model_params: dict[str, Any] = field(default_factory=dict)
    source_meta: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "setting_id": self.setting_id,
            "setting_index": self.setting_index,
            "dataset_name": self.dataset_name,
            "backbone": self.backbone,
            "lookback": self.lookback,
            "horizon": self.horizon,
            "seed": self.seed,
            "train_view_token": self.train_view_token,
            "eval_view_token": self.eval_view_token,
            "train_view_name": self.train_view_name,
            "eval_view_name": self.eval_view_name,
            "eval_row_view_name": self.eval_row_view_name,
            "eval_protocol": self.eval_protocol,
            "subset_name": self.subset_name,
            "apply_eval_intervention": self.apply_eval_intervention,
            "runtime_cfg": dict(self.runtime_cfg),
            "model_params": dict(self.model_params),
            "source_meta": dict(self.source_meta),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentSpec":
        return cls(
            task_name=str(payload["task_name"]),
            setting_id=str(payload["setting_id"]),
            setting_index=int(payload["setting_index"]),
            dataset_name=str(payload["dataset_name"]),
            backbone=str(payload["backbone"]),
            lookback=int(payload["lookback"]),
            horizon=int(payload["horizon"]),
            seed=int(payload["seed"]),
            train_view_token=str(payload["train_view_token"]),
            eval_view_token=str(payload["eval_view_token"]),
            train_view_name=str(payload["train_view_name"]),
            eval_view_name=str(payload["eval_view_name"]),
            eval_row_view_name=str(payload["eval_row_view_name"]),
            eval_protocol=str(payload["eval_protocol"]),
            subset_name=str(payload["subset_name"]),
            apply_eval_intervention=bool(payload["apply_eval_intervention"]),
            runtime_cfg=dict(payload.get("runtime_cfg", {})),
            model_params=dict(payload.get("model_params", {})),
            source_meta=dict(payload.get("source_meta", {})),
            extra=dict(payload.get("extra", {})),
        )


@dataclass(frozen=True)
class RunContext:
    config_path: Path
    view_manifest_path: Path | None
    views_dir: Path
    events_path: Path
    registry_path: Path
    task_root: Path
    report_out: Path
    setting_logs_dir: Path


@dataclass
class SettingResult:
    task_name: str
    setting_id: str
    setting_index: int
    status: str
    metrics: dict[str, Any]
    n_train_windows: int
    n_eval_windows: int
    best_val_metric: float | None
    epochs_ran: int | None
    fit_seconds: float | None
    artifact_paths: dict[str, str]
    error_rows_path: str
    result_row: dict[str, Any]
    spec: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "setting_id": self.setting_id,
            "setting_index": self.setting_index,
            "status": self.status,
            "metrics": dict(self.metrics),
            "n_train_windows": self.n_train_windows,
            "n_eval_windows": self.n_eval_windows,
            "best_val_metric": self.best_val_metric,
            "epochs_ran": self.epochs_ran,
            "fit_seconds": self.fit_seconds,
            "artifact_paths": dict(self.artifact_paths),
            "error_rows_path": self.error_rows_path,
            "result_row": dict(self.result_row),
            "spec": dict(self.spec),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SettingResult":
        return cls(
            task_name=str(payload["task_name"]),
            setting_id=str(payload["setting_id"]),
            setting_index=int(payload["setting_index"]),
            status=str(payload["status"]),
            metrics=dict(payload.get("metrics", {})),
            n_train_windows=int(payload.get("n_train_windows", 0)),
            n_eval_windows=int(payload.get("n_eval_windows", 0)),
            best_val_metric=payload.get("best_val_metric"),
            epochs_ran=payload.get("epochs_ran"),
            fit_seconds=payload.get("fit_seconds"),
            artifact_paths=dict(payload.get("artifact_paths", {})),
            error_rows_path=str(payload.get("error_rows_path", "")),
            result_row=dict(payload.get("result_row", {})),
            spec=dict(payload.get("spec", {})),
        )
