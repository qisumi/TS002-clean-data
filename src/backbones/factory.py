from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from torch import nn
import torch

from data.forecasting import resolve_cycle_length
from data.paths import ROOT_DIR


def _load_python_module(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def instantiate_backbone(
    backbone_name: str,
    seq_len: int,
    pred_len: int,
    n_vars: int,
    params: dict[str, Any],
    dataset_name: str,
) -> nn.Module:
    def _as_int_list(value: Any, default: list[int], min_len: int | None = None) -> list[int]:
        if value is None:
            items = list(default)
        elif isinstance(value, (list, tuple)):
            items = [int(item) for item in value]
        else:
            text = str(value).strip()
            items = list(default) if not text else [int(item) for item in text.replace(",", " ").split()]
        if min_len is not None and items and len(items) < min_len:
            items = items + [items[-1]] * (min_len - len(items))
        return items

    if backbone_name == "DLinear":
        module = _load_python_module(
            module_name="baseline_dlinear_module",
            module_path=ROOT_DIR / "baseline" / "DLinear" / "DLinear.py",
        )
        cfg = SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=n_vars,
            individual=bool(params.get("individual", False)),
        )
        model = module.Model(cfg)
        setattr(model, "_backbone_name", backbone_name)
        return model

    if backbone_name == "PatchTST":
        patch_root = ROOT_DIR / "baseline" / "PatchTST"
        if str(patch_root) not in sys.path:
            sys.path.insert(0, str(patch_root))
        module = importlib.import_module("PatchTST")
        cfg = SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=n_vars,
            individual=bool(params.get("individual", False)),
            e_layers=int(params.get("e_layers", 3)),
            n_heads=int(params.get("n_heads", 4)),
            d_model=int(params.get("d_model", 64)),
            d_ff=int(params.get("d_ff", 128)),
            dropout=float(params.get("dropout", 0.1)),
            fc_dropout=float(params.get("fc_dropout", 0.1)),
            head_dropout=float(params.get("head_dropout", 0.0)),
            patch_len=int(params.get("patch_len", 16)),
            stride=int(params.get("stride", 8)),
            padding_patch=str(params.get("padding_patch", "end")),
            revin=bool(params.get("revin", True)),
            affine=bool(params.get("affine", True)),
            subtract_last=bool(params.get("subtract_last", False)),
            decomposition=bool(params.get("decomposition", False)),
            kernel_size=int(params.get("kernel_size", 25)),
        )
        model = module.Model(cfg)
        setattr(model, "_backbone_name", backbone_name)
        return model

    if backbone_name == "TQNet":
        module = _load_python_module(
            module_name="baseline_tqnet_module",
            module_path=ROOT_DIR / "baseline" / "TQNet" / "TQNet.py",
        )
        cfg = SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=n_vars,
            cycle=resolve_cycle_length(dataset_name, params),
            model_type=str(params.get("model_type", "TQNet")),
            d_model=int(params.get("d_model", 64)),
            dropout=float(params.get("dropout", 0.1)),
            use_revin=bool(params.get("use_revin", True)),
        )
        model = module.Model(cfg)
        setattr(model, "_backbone_name", backbone_name)
        setattr(model, "_cycle_len", int(cfg.cycle))
        return model

    if backbone_name == "iTransformer":
        module = _load_python_module(
            module_name="baseline_itransformer_module",
            module_path=ROOT_DIR / "baseline" / "iTransformer" / "iTransformer.py",
        )
        cfg = SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=n_vars,
            d_model=int(params.get("d_model", 512)),
            n_heads=int(params.get("n_heads", 8)),
            e_layers=int(params.get("e_layers", 2)),
            d_ff=int(params.get("d_ff", 2048)),
            factor=int(params.get("factor", 1)),
            dropout=float(params.get("dropout", 0.1)),
            activation=str(params.get("activation", "gelu")),
            use_norm=bool(params.get("use_norm", True)),
        )
        model = module.Model(cfg)
        setattr(model, "_backbone_name", backbone_name)
        return model

    if backbone_name == "ModernTCN":
        module = _load_python_module(
            module_name="baseline_moderntcn_module",
            module_path=ROOT_DIR / "baseline" / "ModernTCN" / "ModernTCN.py",
        )
        num_blocks = _as_int_list(params.get("num_blocks"), [1])
        num_stages = len(num_blocks)
        cfg = SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=n_vars,
            patch_size=int(params.get("patch_size", 8)),
            patch_stride=int(params.get("patch_stride", 4)),
            downsample_ratio=int(params.get("downsample_ratio", 2)),
            ffn_ratio=int(params.get("ffn_ratio", 1)),
            num_blocks=num_blocks,
            large_size=_as_int_list(params.get("large_size"), [51], min_len=num_stages),
            small_size=_as_int_list(params.get("small_size"), [5], min_len=num_stages),
            dims=_as_int_list(params.get("dims"), [64], min_len=num_stages),
            dw_dims=_as_int_list(params.get("dw_dims"), [64], min_len=num_stages),
            head_dropout=float(params.get("head_dropout", 0.0)),
            dropout=float(params.get("dropout", 0.3)),
            revin=bool(params.get("revin", True)),
            affine=bool(params.get("affine", False)),
            subtract_last=bool(params.get("subtract_last", False)),
            decomposition=bool(params.get("decomposition", False)),
            kernel_size=int(params.get("kernel_size", 25)),
        )
        model = module.Model(cfg)
        setattr(model, "_backbone_name", backbone_name)
        return model

    if backbone_name == "TimeMixer":
        module = _load_python_module(
            module_name="baseline_timemixer_module",
            module_path=ROOT_DIR / "baseline" / "TimeMixer" / "TimeMixer.py",
        )
        cfg = SimpleNamespace(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=n_vars,
            c_out=n_vars,
            e_layers=int(params.get("e_layers", 2)),
            d_model=int(params.get("d_model", 16)),
            d_ff=int(params.get("d_ff", 32)),
            dropout=float(params.get("dropout", 0.1)),
            moving_avg=int(params.get("moving_avg", 25)),
            channel_independence=int(params.get("channel_independence", 1)),
            decomp_method=str(params.get("decomp_method", "moving_avg")),
            top_k=int(params.get("top_k", 5)),
            use_norm=int(bool(params.get("use_norm", True))),
            down_sampling_layers=int(params.get("down_sampling_layers", 3)),
            down_sampling_window=int(params.get("down_sampling_window", 2)),
            down_sampling_method=str(params.get("down_sampling_method", "avg")),
        )
        model = module.Model(cfg)
        setattr(model, "_backbone_name", backbone_name)
        return model

    if backbone_name == "TimeMixerPP":
        module = _load_python_module(
            module_name="baseline_timemixerpp_module",
            module_path=ROOT_DIR / "baseline" / "TimeMixerPP" / "TimeMixerPP.py",
        )
        cfg = module.TimeMixerPPConfig(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=n_vars,
            d_model=int(params.get("d_model", 256)),
            expert_hidden=int(params.get("expert_hidden", 384)),
            head_rank=int(params.get("head_rank", 32)),
            patch_len=int(params.get("patch_len", 8)),
            patch_stride=int(params.get("patch_stride", 4)),
            n_blocks=int(params.get("n_blocks", 4)),
            n_resolutions=int(params.get("n_resolutions", 3)),
            n_heads=int(params.get("n_heads", 8)),
            ffn_ratio=int(params.get("ffn_ratio", 4)),
            dropout=float(params.get("dropout", 0.1)),
            stochastic_depth=float(params.get("stochastic_depth", 0.1)),
            num_experts=int(params.get("num_experts", 4)),
            horizon_vocab_size=1,
        )
        model = module.Model(cfg)
        setattr(model, "_backbone_name", backbone_name)
        return model

    raise ValueError(f"Unsupported backbone: {backbone_name}")


def forward_backbone(model: nn.Module, x: torch.Tensor, cycle_index: torch.Tensor | None = None) -> torch.Tensor:
    backbone_name = str(getattr(model, "_backbone_name", ""))
    if backbone_name == "TQNet":
        if cycle_index is None:
            raise ValueError("TQNet forward requires cycle_index")
        return model(x, cycle_index)
    return model(x)
