from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


"""Dataset-specific presets for the refactored AIFPlus / AEFPlus models.

These presets are performance-oriented. By default we keep the current config
lookback unless `defaults.use_preset_lookback=true`, so existing 96-lookback
pipelines continue to run without regenerating view windows.
"""


AIF_PRESETS: dict[str, dict[str, Any]] = {
    "ETTh1": {
        "seq_len": 96,
        "d_model": 256,
        "n_blocks": 4,
        "n_heads": 8,
        "ffn_ratio": 4,
        "dropout": 0.10,
        "use_diff_branch": False,
        "patch_len_small": 8,
        "patch_stride_small": 4,
        "patch_len_large": 16,
        "patch_stride_large": 8,
        "periods": (24, 48, 96),
        "residual_hidden": 16,
        "lambda_res_max": 0.003,
        "batch_size": 128,
        "learning_rate": 2e-4,
        "weight_decay": 0.05,
        "epochs": 60,
        "patience": 10,
    },
    "ETTh2": {
        "seq_len": 96,
        "d_model": 256,
        "n_blocks": 4,
        "n_heads": 8,
        "ffn_ratio": 4,
        "dropout": 0.12,
        "use_diff_branch": False,
        "patch_len_small": 8,
        "patch_stride_small": 4,
        "patch_len_large": 16,
        "patch_stride_large": 8,
        "periods": (24, 48, 96),
        "residual_hidden": 24,
        "lambda_res_max": 0.008,
        "batch_size": 128,
        "learning_rate": 2e-4,
        "weight_decay": 0.05,
        "epochs": 60,
        "patience": 10,
    },
    "ETTm1": {
        "seq_len": 96,
        "d_model": 320,
        "n_blocks": 4,
        "n_heads": 8,
        "ffn_ratio": 4,
        "dropout": 0.12,
        "use_diff_branch": True,
        "patch_len_small": 8,
        "patch_stride_small": 4,
        "patch_len_large": 16,
        "patch_stride_large": 8,
        "periods": (24, 48, 96),
        "residual_hidden": 16,
        "lambda_res_max": 0.004,
        "batch_size": 128,
        "learning_rate": 1.5e-4,
        "weight_decay": 0.05,
        "epochs": 60,
        "patience": 10,
    },
    "ETTm2": {
        "seq_len": 96,
        "d_model": 320,
        "n_blocks": 4,
        "n_heads": 8,
        "ffn_ratio": 4,
        "dropout": 0.15,
        "use_diff_branch": True,
        "patch_len_small": 8,
        "patch_stride_small": 4,
        "patch_len_large": 16,
        "patch_stride_large": 8,
        "periods": (24, 48, 96),
        "residual_hidden": 24,
        "lambda_res_max": 0.010,
        "batch_size": 128,
        "learning_rate": 1.5e-4,
        "weight_decay": 0.05,
        "epochs": 60,
        "patience": 10,
    },
    "solar_AL": {
        "seq_len": 96,
        "d_model": 384,
        "n_blocks": 5,
        "n_heads": 8,
        "ffn_ratio": 4,
        "dropout": 0.12,
        "use_diff_branch": True,
        "patch_len_small": 8,
        "patch_stride_small": 4,
        "patch_len_large": 16,
        "patch_stride_large": 8,
        "periods": (24, 48, 96),
        "residual_hidden": 32,
        "lambda_res_max": 0.015,
        "batch_size": 64,
        "learning_rate": 4e-4,
        "weight_decay": 0.05,
        "epochs": 50,
        "patience": 8,
    },
    "weather": {
        "seq_len": 96,
        "d_model": 384,
        "n_blocks": 5,
        "n_heads": 8,
        "ffn_ratio": 4,
        "dropout": 0.15,
        "use_diff_branch": True,
        "patch_len_small": 8,
        "patch_stride_small": 4,
        "patch_len_large": 16,
        "patch_stride_large": 8,
        "periods": (24, 48, 96),
        "residual_hidden": 24,
        "lambda_res_max": 0.010,
        "batch_size": 64,
        "learning_rate": 2.5e-4,
        "weight_decay": 0.05,
        "epochs": 50,
        "patience": 8,
    },
    "exchange_rate": {
        "seq_len": 96,
        "d_model": 128,
        "n_blocks": 3,
        "n_heads": 4,
        "ffn_ratio": 4,
        "dropout": 0.20,
        "use_diff_branch": False,
        "patch_len_small": 8,
        "patch_stride_small": 4,
        "patch_len_large": 16,
        "patch_stride_large": 8,
        "periods": (7, 28, 56),
        "residual_hidden": 8,
        "lambda_res_max": 0.0,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "weight_decay": 0.05,
        "epochs": 80,
        "patience": 12,
    },
    "electricity": {
        "seq_len": 96,
        "d_model": 384,
        "n_blocks": 5,
        "n_heads": 8,
        "ffn_ratio": 4,
        "dropout": 0.15,
        "use_diff_branch": True,
        "patch_len_small": 8,
        "patch_stride_small": 4,
        "patch_len_large": 16,
        "patch_stride_large": 8,
        "periods": (24, 48, 96),
        "residual_hidden": 24,
        "lambda_res_max": 0.010,
        "batch_size": 16,
        "learning_rate": 4e-4,
        "weight_decay": 0.05,
        "epochs": 50,
        "patience": 8,
    },
}


AEF_PRESETS: dict[str, dict[str, Any]] = {
    "ETTh1": {
        "seq_len": 336,
        "enc_in": 7,
        "d_model": 160,
        "metadata_dim": 24,
        "expert_hidden": 224,
        "head_rank": 24,
        "patch_len": 16,
        "patch_stride": 8,
        "n_blocks": 3,
        "n_heads": 4,
        "ffn_ratio": 3,
        "dropout": 0.10,
        "stochastic_depth": 0.05,
        "num_experts": 3,
        "max_boundary_steps": 96,
        "batch_size": 64,
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "epochs": 30,
        "patience": 6,
    },
    "ETTh2": {
        "seq_len": 336,
        "enc_in": 7,
        "d_model": 128,
        "metadata_dim": 24,
        "expert_hidden": 192,
        "head_rank": 24,
        "patch_len": 16,
        "patch_stride": 8,
        "n_blocks": 3,
        "n_heads": 4,
        "ffn_ratio": 3,
        "dropout": 0.12,
        "stochastic_depth": 0.05,
        "num_experts": 3,
        "max_boundary_steps": 96,
        "batch_size": 64,
        "learning_rate": 8e-4,
        "weight_decay": 2e-4,
        "epochs": 30,
        "patience": 6,
    },
    "ETTm1": {
        "seq_len": 384,
        "enc_in": 7,
        "d_model": 160,
        "metadata_dim": 24,
        "expert_hidden": 256,
        "head_rank": 24,
        "patch_len": 24,
        "patch_stride": 12,
        "n_blocks": 4,
        "n_heads": 4,
        "ffn_ratio": 3,
        "dropout": 0.10,
        "stochastic_depth": 0.08,
        "num_experts": 4,
        "max_boundary_steps": 96,
        "batch_size": 32,
        "learning_rate": 6e-4,
        "weight_decay": 1e-4,
        "epochs": 24,
        "patience": 5,
    },
    "ETTm2": {
        "seq_len": 384,
        "enc_in": 7,
        "d_model": 160,
        "metadata_dim": 24,
        "expert_hidden": 256,
        "head_rank": 24,
        "patch_len": 24,
        "patch_stride": 12,
        "n_blocks": 4,
        "n_heads": 4,
        "ffn_ratio": 3,
        "dropout": 0.10,
        "stochastic_depth": 0.08,
        "num_experts": 4,
        "max_boundary_steps": 96,
        "batch_size": 32,
        "learning_rate": 6e-4,
        "weight_decay": 1e-4,
        "epochs": 24,
        "patience": 5,
    },
    "solar_AL": {
        "seq_len": 384,
        "enc_in": 137,
        "d_model": 224,
        "metadata_dim": 32,
        "expert_hidden": 384,
        "head_rank": 32,
        "patch_len": 24,
        "patch_stride": 12,
        "n_blocks": 4,
        "n_heads": 8,
        "ffn_ratio": 4,
        "dropout": 0.10,
        "stochastic_depth": 0.10,
        "num_experts": 4,
        "max_boundary_steps": 96,
        "batch_size": 24,
        "learning_rate": 4e-4,
        "weight_decay": 1e-4,
        "epochs": 20,
        "patience": 4,
    },
    "weather": {
        "seq_len": 384,
        "enc_in": 21,
        "d_model": 192,
        "metadata_dim": 32,
        "expert_hidden": 320,
        "head_rank": 32,
        "patch_len": 24,
        "patch_stride": 12,
        "n_blocks": 4,
        "n_heads": 8,
        "ffn_ratio": 4,
        "dropout": 0.10,
        "stochastic_depth": 0.08,
        "num_experts": 4,
        "max_boundary_steps": 96,
        "batch_size": 32,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "epochs": 20,
        "patience": 4,
    },
    "exchange_rate": {
        "seq_len": 192,
        "enc_in": 8,
        "d_model": 128,
        "metadata_dim": 16,
        "expert_hidden": 192,
        "head_rank": 16,
        "patch_len": 8,
        "patch_stride": 4,
        "n_blocks": 2,
        "n_heads": 4,
        "ffn_ratio": 3,
        "dropout": 0.15,
        "stochastic_depth": 0.03,
        "num_experts": 3,
        "max_boundary_steps": 96,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 5e-4,
        "epochs": 40,
        "patience": 8,
    },
    "electricity": {
        "seq_len": 336,
        "enc_in": 321,
        "d_model": 256,
        "metadata_dim": 32,
        "expert_hidden": 448,
        "head_rank": 32,
        "patch_len": 24,
        "patch_stride": 12,
        "n_blocks": 4,
        "n_heads": 8,
        "ffn_ratio": 4,
        "dropout": 0.08,
        "stochastic_depth": 0.10,
        "num_experts": 4,
        "max_boundary_steps": 96,
        "batch_size": 16,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "epochs": 18,
        "patience": 4,
    },
}


COMMON_TRAINING: dict[str, Any] = {
    "optimizer": "AdamW",
    "betas": (0.9, 0.95),
    "grad_clip": 1.0,
    "amp": True,
    "scheduler": "cosine",
    "warmup_ratio": 0.1,
    "label_free_aux": {
        "AIF": {
            "lambda_reconstruction": 0.1,
            "lambda_router_balance": 0.001,
            "lambda_artifact_adv": 0.0,
            "lambda_phase_adv": 0.0,
            "lambda_artifact_aux": 0.0,
            "lambda_phase_aux": 0.0,
            "grl_alpha_final": 0.0,
            "grl_warmup_ratio": 0.0,
        },
        "AEF": {
            "lambda_router_balance": 0.001,
            "lambda_group": 0.0,
            "lambda_phase": 0.0,
            "lambda_severity": 0.0,
        },
    },
    "public_benchmark": {
        "AIF": {
            "lambda_reconstruction": 0.05,
            "lambda_router_balance": 0.001,
            "lambda_artifact_adv": 0.0,
            "lambda_phase_adv": 0.0,
            "lambda_artifact_aux": 0.01,
            "lambda_phase_aux": 0.01,
            "grl_alpha_final": 0.0,
            "grl_warmup_ratio": 0.0,
        },
        "AEF": {
            "lambda_router_balance": 0.001,
            "lambda_group": 0.0,
            "lambda_phase": 0.0,
            "lambda_severity": 0.0,
        },
    },
    "with_aux_labels": {
        "AIF": {
            "lambda_reconstruction": 0.3,
            "lambda_router_balance": 0.01,
            "lambda_artifact_adv": 0.05,
            "lambda_phase_adv": 0.05,
            "lambda_artifact_aux": 0.10,
            "lambda_phase_aux": 0.10,
            "grl_alpha_final": 1.0,
            "grl_warmup_ratio": 0.3,
        },
        "AEF": {
            "lambda_router_balance": 0.01,
            "lambda_group": 0.05,
            "lambda_phase": 0.05,
            "lambda_severity": 0.02,
        },
    },
}


def _copy_dict(value: dict[str, Any] | None) -> dict[str, Any]:
    return deepcopy(value or {})


def _merge_nested_dict(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    if not override:
        return base
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge_nested_dict(_copy_dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base


def _resolve_aux_label_mode(defaults: dict[str, Any]) -> str:
    mode = str(defaults.get("aux_label_mode", "with_aux_labels")).strip() or "with_aux_labels"
    if mode not in {"label_free_aux", "public_benchmark", "with_aux_labels"}:
        raise ValueError(f"Unsupported aux_label_mode `{mode}`")
    return mode


def _default_eval_batch_size(batch_size: int) -> int:
    return max(int(batch_size), int(batch_size) * 2)


def _apply_common_runtime(runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    runtime_cfg = _copy_dict(runtime_cfg)
    runtime_cfg.setdefault("betas", tuple(float(value) for value in COMMON_TRAINING["betas"]))
    runtime_cfg.setdefault("grad_clip", float(COMMON_TRAINING["grad_clip"]))
    runtime_cfg.setdefault("amp", bool(COMMON_TRAINING["amp"]))
    runtime_cfg.setdefault("warmup_ratio", float(COMMON_TRAINING["warmup_ratio"]))
    runtime_cfg.setdefault("optimizer", str(COMMON_TRAINING["optimizer"]))
    runtime_cfg.setdefault("scheduler", str(COMMON_TRAINING["scheduler"]))
    return runtime_cfg


def _proportional_ints(total: int, weights: list[int]) -> list[int]:
    if total <= 0 or not weights:
        return [0 for _ in weights]

    positive_indices = [idx for idx, weight in enumerate(weights) if weight > 0]
    if not positive_indices:
        return [0 for _ in weights]

    counts = [0 for _ in weights]
    if total >= len(positive_indices):
        for idx in positive_indices:
            counts[idx] = 1
        remaining = total - len(positive_indices)
    else:
        ranked = sorted(positive_indices, key=lambda idx: weights[idx], reverse=True)
        for idx in ranked[:total]:
            counts[idx] = 1
        return counts

    total_weight = sum(weights[idx] for idx in positive_indices)
    fractional: list[tuple[float, int]] = []
    for idx in positive_indices:
        raw = 0.0 if total_weight <= 0 else remaining * float(weights[idx]) / float(total_weight)
        floor_value = int(math.floor(raw))
        counts[idx] += floor_value
        fractional.append((raw - float(floor_value), idx))

    leftover = remaining - sum(max(counts[idx] - 1, 0) for idx in positive_indices)
    for _, idx in sorted(fractional, reverse=True):
        if leftover <= 0:
            break
        counts[idx] += 1
        leftover -= 1
    return counts


def _scale_aif_stages(
    base_runtime_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    base_stage_cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    stages = _copy_dict(base_stage_cfg)
    if not stages:
        return stages

    total_epochs = int(runtime_cfg.get("epochs", 0))
    total_patience = int(runtime_cfg.get("patience", 0))
    if total_epochs <= 0:
        return stages
    stage_names = [name for name in ["stage_a", "stage_b", "stage_c"] if name in stages]
    if not stage_names:
        return stages

    epoch_weights = [max(int(stages[name].get("epochs", 0)), 0) for name in stage_names]
    scaled_epochs = _proportional_ints(total_epochs, epoch_weights)

    base_runtime_patience = max(1, int(base_runtime_cfg.get("patience", total_patience if total_patience > 0 else 1)))
    base_runtime_lr = float(base_runtime_cfg.get("lr", runtime_cfg.get("lr", 0.0)) or runtime_cfg.get("lr", 0.0) or 0.0)
    target_lr = float(runtime_cfg.get("lr", base_runtime_lr or 0.0))

    for stage_name, stage_epochs in zip(stage_names, scaled_epochs):
        stage_cfg = stages[stage_name]
        base_stage_lr = float(stage_cfg.get("lr", base_runtime_lr or target_lr))
        lr_factor = 1.0 if base_runtime_lr <= 0.0 else base_stage_lr / base_runtime_lr
        stage_cfg["epochs"] = int(stage_epochs)
        stage_cfg["lr"] = float(target_lr * lr_factor)

        base_stage_patience = max(1, int(stage_cfg.get("patience", base_runtime_patience)))
        scaled_patience = max(1, int(round(float(total_patience) * float(base_stage_patience) / float(base_runtime_patience))))
        stage_cfg["patience"] = int(min(max(stage_epochs, 1), scaled_patience))
    return stages


def _preset_source_note(model_name: str, use_preset_lookback: bool) -> str:
    if use_preset_lookback:
        return f"{model_name} dataset preset with dataset-specific seq_len/lookback."
    return f"{model_name} dataset preset with current config lookback preserved."


def resolve_aif_plus_dataset_config(defaults: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    runtime_cfg = _copy_dict(defaults.get("runtime"))
    model_cfg = _copy_dict(defaults.get("model"))
    loss_cfg = _copy_dict(defaults.get("loss"))
    stage_cfg = _copy_dict(defaults.get("stages"))

    use_dataset_presets = bool(defaults.get("use_dataset_presets", False))
    use_preset_lookback = bool(defaults.get("use_preset_lookback", False))
    prefer_config_model_over_presets = bool(defaults.get("prefer_config_model_over_presets", False))
    aux_label_mode = _resolve_aux_label_mode(defaults)
    lookback = int(defaults.get("lookback", 96))

    source_kind = "config_default"
    source_url = "configs/aif_plus.yaml"
    source_note = "AIF-Plus default config"

    if use_dataset_presets:
        preset = AIF_PRESETS.get(dataset_name)
        if preset is not None:
            if use_preset_lookback:
                lookback = int(preset["seq_len"])
            preset_model_cfg = {
                key: deepcopy(preset[key])
                for key in (
                    "d_model",
                    "n_patch_layers",
                    "n_decoder_layers",
                    "n_blocks",
                    "n_heads",
                    "ffn_ratio",
                    "dropout",
                    "use_diff_branch",
                    "patch_len",
                    "patch_stride",
                    "patch_len_small",
                    "patch_stride_small",
                    "patch_len_large",
                    "patch_stride_large",
                    "patch_jitter",
                    "periods",
                    "query_period",
                    "queries_per_period",
                    "spectral_topk",
                    "residual_hidden",
                    "lambda_res_max",
                )
                if key in preset and preset[key] is not None
            }
            if prefer_config_model_over_presets:
                for key, value in preset_model_cfg.items():
                    if value is not None:
                        model_cfg.setdefault(key, value)
            else:
                model_cfg.update({key: value for key, value in preset_model_cfg.items() if value is not None})
            runtime_cfg.update(
                {
                    "batch_size": preset["batch_size"],
                    "eval_batch_size": _default_eval_batch_size(int(preset["batch_size"])),
                    "lr": preset["learning_rate"],
                    "weight_decay": preset["weight_decay"],
                    "epochs": preset["epochs"],
                    "patience": preset["patience"],
                }
            )
            source_kind = "dataset_hparam_preset"
            source_url = "src/utils/dataset_hparam_presets.py"
            source_note = _preset_source_note("AIF-Plus", use_preset_lookback)
            if prefer_config_model_over_presets:
                source_note += " Explicit model keys in config take priority over preset model values."

    dataset_overrides = defaults.get("dataset_overrides", {})
    if isinstance(dataset_overrides, dict):
        dataset_override = dataset_overrides.get(dataset_name)
        if isinstance(dataset_override, dict):
            if "lookback" in dataset_override:
                lookback = int(dataset_override["lookback"])
            runtime_cfg = _merge_nested_dict(runtime_cfg, dataset_override.get("runtime"))
            model_cfg = _merge_nested_dict(model_cfg, dataset_override.get("model"))
            loss_cfg = _merge_nested_dict(loss_cfg, dataset_override.get("loss"))
            stage_cfg = _merge_nested_dict(stage_cfg, dataset_override.get("stages"))
            source_note += " Applied dataset_overrides from config."

    base_runtime_cfg = _copy_dict(runtime_cfg)
    runtime_cfg = _apply_common_runtime(runtime_cfg)
    aux_cfg = _copy_dict(COMMON_TRAINING[aux_label_mode]["AIF"])
    loss_cfg.setdefault("gamma_rec", float(aux_cfg.pop("lambda_reconstruction")))
    for key, value in aux_cfg.items():
        loss_cfg.setdefault(key, value)
    stage_cfg = _scale_aif_stages(base_runtime_cfg=base_runtime_cfg, runtime_cfg=runtime_cfg, base_stage_cfg=stage_cfg)

    return {
        "lookback": int(lookback),
        "runtime": runtime_cfg,
        "model": model_cfg,
        "loss": loss_cfg,
        "stages": stage_cfg,
        "aux_label_mode": aux_label_mode,
        "collapse_aux_label_vocabs": aux_label_mode == "label_free_aux",
        "hyperparam_source_kind": source_kind,
        "hyperparam_source_url": source_url,
        "hyperparam_source_note": source_note,
    }


def resolve_aef_plus_dataset_config(defaults: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    runtime_cfg = _copy_dict(defaults.get("runtime"))
    model_cfg = _copy_dict(defaults.get("model"))
    loss_cfg = _copy_dict(defaults.get("loss"))

    use_dataset_presets = bool(defaults.get("use_dataset_presets", False))
    use_preset_lookback = bool(defaults.get("use_preset_lookback", False))
    aux_label_mode = _resolve_aux_label_mode(defaults)
    lookback = int(defaults.get("lookback", 96))

    source_kind = "config_default"
    source_url = "configs/aef_plus.yaml"
    source_note = "AEF-Plus default config"

    if use_dataset_presets:
        preset = AEF_PRESETS.get(dataset_name)
        if preset is not None:
            if use_preset_lookback:
                lookback = int(preset["seq_len"])
            model_cfg.update(
                {
                    "d_model": preset["d_model"],
                    "metadata_dim": preset["metadata_dim"],
                    "expert_hidden": preset["expert_hidden"],
                    "head_rank": preset["head_rank"],
                    "patch_len": preset["patch_len"],
                    "patch_stride": preset["patch_stride"],
                    "n_blocks": preset["n_blocks"],
                    "n_heads": preset["n_heads"],
                    "ffn_ratio": preset["ffn_ratio"],
                    "dropout": preset["dropout"],
                    "stochastic_depth": preset["stochastic_depth"],
                    "num_experts": preset["num_experts"],
                    "max_boundary_steps": preset["max_boundary_steps"],
                }
            )
            runtime_cfg.update(
                {
                    "batch_size": preset["batch_size"],
                    "eval_batch_size": _default_eval_batch_size(int(preset["batch_size"])),
                    "lr": preset["learning_rate"],
                    "weight_decay": preset["weight_decay"],
                    "epochs": preset["epochs"],
                    "patience": preset["patience"],
                }
            )
            source_kind = "dataset_hparam_preset"
            source_url = "src/utils/dataset_hparam_presets.py"
            source_note = _preset_source_note("AEF-Plus", use_preset_lookback)

    runtime_cfg = _apply_common_runtime(runtime_cfg)
    loss_cfg.update(_copy_dict(COMMON_TRAINING[aux_label_mode]["AEF"]))
    model_cfg["max_boundary_steps"] = max(96, int(model_cfg.get("max_boundary_steps", 96)))

    return {
        "lookback": int(lookback),
        "runtime": runtime_cfg,
        "model": model_cfg,
        "loss": loss_cfg,
        "aux_label_mode": aux_label_mode,
        "collapse_aux_label_vocabs": aux_label_mode == "label_free_aux",
        "hyperparam_source_kind": source_kind,
        "hyperparam_source_url": source_url,
        "hyperparam_source_note": source_note,
    }
