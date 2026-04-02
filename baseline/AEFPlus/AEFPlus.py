from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

_BASELINE_DIR = Path(__file__).resolve().parents[1]
if str(_BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASELINE_DIR))

from ts_refactor_common import (
    AttentionBlock,
    ClassifierHead,
    GlobalFusion,
    RegressionHead,
    RevIN,
    ScaleFrequencyEncoder,
    TailBoundaryEncoder,
    TemporalPatchEncoder,
    TopKRouter,
    VariableAwareForecastHead,
    VariateTokenEncoder,
)


@dataclass
class AEFPlusConfig:
    seq_len: int
    pred_len: int
    enc_in: int
    artifact_vocab_size: int = 1
    phase_vocab_size: int = 1
    severity_vocab_size: int = 1
    nvar_vocab_size: int = 1
    horizon_vocab_size: int = 1
    metadata_num_dim: int = 8
    d_model: int = 192
    metadata_dim: int = 32
    expert_hidden: int = 256
    head_rank: int = 24
    patch_len: int = 8
    patch_stride: int = 4
    n_blocks: int = 3
    n_heads: int = 4
    ffn_ratio: int = 3
    dropout: float = 0.1
    stochastic_depth: float = 0.05
    num_experts: int = 4
    max_boundary_steps: int = 96




def _compatible_heads(requested_heads: int, d_model: int) -> int:
    requested_heads = int(max(1, requested_heads))
    d_model = int(max(1, d_model))
    for heads in range(min(requested_heads, d_model), 0, -1):
        if d_model % heads == 0:
            return heads
    return 1


class AEFMetadataEncoder(nn.Module):
    def __init__(
        self,
        metadata_num_dim: int,
        metadata_hidden: int,
        d_model: int,
        artifact_vocab_size: int,
        phase_vocab_size: int,
        severity_vocab_size: int,
        nvar_vocab_size: int,
        horizon_vocab_size: int,
        dropout: float,
        n_heads: int,
    ) -> None:
        super().__init__()
        hidden = max(int(metadata_hidden) * 2, d_model // 2)
        self.numeric_mlp = nn.Sequential(
            nn.LayerNorm(metadata_num_dim),
            nn.Linear(metadata_num_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.artifact_embed = nn.Embedding(max(int(artifact_vocab_size), 1), d_model)
        self.phase_embed = nn.Embedding(max(int(phase_vocab_size), 1), d_model)
        self.severity_embed = nn.Embedding(max(int(severity_vocab_size), 1), d_model)
        self.nvar_embed = nn.Embedding(max(int(nvar_vocab_size), 1), d_model)
        self.horizon_embed = nn.Embedding(max(int(horizon_vocab_size), 1), d_model)
        self.positional = nn.Parameter(torch.zeros(1, 6, d_model))
        self.layers = nn.ModuleList(
            [
                AttentionBlock(
                    d_model=d_model,
                    n_heads=_compatible_heads(max(1, n_heads), d_model),
                    ffn_ratio=2,
                    dropout=dropout,
                    drop_path=0.0,
                )
                for _ in range(2)
            ]
        )
        self.summary = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        metadata_num: torch.Tensor,
        artifact_id: torch.Tensor,
        phase_id: torch.Tensor,
        severity_bin_id: torch.Tensor,
        nvar_bin_id: torch.Tensor,
        horizon_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack(
            [
                self.numeric_mlp(metadata_num),
                self.artifact_embed(artifact_id),
                self.phase_embed(phase_id),
                self.severity_embed(severity_bin_id),
                self.nvar_embed(nvar_bin_id),
                self.horizon_embed(horizon_id),
            ],
            dim=1,
        )
        tokens = tokens + self.positional
        for layer in self.layers:
            tokens = layer(tokens)
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)
        return tokens, self.summary(pooled)


class AEFBackbone(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_vars: int,
        d_model: int,
        patch_len: int,
        patch_stride: int,
        n_blocks: int,
        n_heads: int,
        ffn_ratio: int,
        dropout: float,
        stochastic_depth: float,
        max_boundary_steps: int,
        metadata_num_dim: int,
        metadata_dim: int,
        artifact_vocab_size: int,
        phase_vocab_size: int,
        severity_vocab_size: int,
        nvar_vocab_size: int,
        horizon_vocab_size: int,
    ) -> None:
        super().__init__()
        self.temporal_encoder = TemporalPatchEncoder(
            input_dim=n_vars,
            seq_len=seq_len,
            d_model=d_model,
            patch_len=patch_len,
            patch_stride=patch_stride,
            n_layers=max(2, n_blocks),
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            stochastic_depth=stochastic_depth,
        )
        self.variate_encoder = VariateTokenEncoder(
            seq_len=seq_len,
            n_vars=n_vars,
            per_var_feature_dim=1,
            d_model=d_model,
            n_heads=_compatible_heads(min(max(1, n_heads), max(1, n_vars)), d_model),
            n_layers=max(2, n_blocks // 2 + 1),
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            stochastic_depth=stochastic_depth,
        )
        self.scale_encoder = ScaleFrequencyEncoder(
            n_vars=n_vars,
            seq_len=seq_len,
            d_model=d_model,
            dropout=dropout,
            scales=(1, 2, 4),
            spectral_topk=8,
        )
        self.boundary_encoder = TailBoundaryEncoder(
            input_dim=n_vars,
            d_model=d_model,
            max_steps=max_boundary_steps,
            dropout=dropout,
            n_layers=2,
        )
        self.meta_encoder = AEFMetadataEncoder(
            metadata_num_dim=metadata_num_dim,
            metadata_hidden=metadata_dim,
            d_model=d_model,
            artifact_vocab_size=artifact_vocab_size,
            phase_vocab_size=phase_vocab_size,
            severity_vocab_size=severity_vocab_size,
            nvar_vocab_size=nvar_vocab_size,
            horizon_vocab_size=horizon_vocab_size,
            dropout=dropout,
            n_heads=n_heads,
        )
        self.type_embedding = nn.Parameter(torch.zeros(5, d_model))
        self.seed_proj = nn.Sequential(
            nn.LayerNorm(d_model * 5),
            nn.Linear(d_model * 5, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = GlobalFusion(
            d_model=d_model,
            n_heads=_compatible_heads(max(1, n_heads), d_model),
            n_layers=max(2, n_blocks // 2 + 1),
            dropout=dropout,
            stochastic_depth=stochastic_depth,
        )

    def forward(
        self,
        x_norm: torch.Tensor,
        metadata_num: torch.Tensor,
        artifact_id: torch.Tensor,
        phase_id: torch.Tensor,
        severity_bin_id: torch.Tensor,
        nvar_bin_id: torch.Tensor,
        horizon_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        temp_tokens, temp_state = self.temporal_encoder(x_norm)
        var_tokens, var_state = self.variate_encoder(x_norm.unsqueeze(-1))
        scale_tokens, scale_state = self.scale_encoder(x_norm)
        boundary_tokens, boundary_state = self.boundary_encoder(x_norm)
        meta_tokens, meta_state = self.meta_encoder(
            metadata_num=metadata_num,
            artifact_id=artifact_id,
            phase_id=phase_id,
            severity_bin_id=severity_bin_id,
            nvar_bin_id=nvar_bin_id,
            horizon_id=horizon_id,
        )

        temp_tokens = temp_tokens + self.type_embedding[0].view(1, 1, -1)
        var_tokens = var_tokens + self.type_embedding[1].view(1, 1, -1)
        scale_tokens = scale_tokens + self.type_embedding[2].view(1, 1, -1)
        boundary_tokens = boundary_tokens + self.type_embedding[3].view(1, 1, -1)
        meta_tokens = meta_tokens + self.type_embedding[4].view(1, 1, -1)

        seed = self.seed_proj(
            torch.cat([temp_state, var_state, scale_state, boundary_state, meta_state], dim=-1)
        )
        state = self.fusion(
            seed=seed,
            context=torch.cat([temp_tokens, var_tokens, scale_tokens, boundary_tokens, meta_tokens], dim=1),
        )
        return {
            "state": state,
            "temp_tokens": temp_tokens,
            "temp_state": temp_state,
            "var_tokens": var_tokens,
            "var_state": var_state,
            "scale_tokens": scale_tokens,
            "scale_state": scale_state,
            "boundary_tokens": boundary_tokens,
            "boundary_state": boundary_state,
            "meta_tokens": meta_tokens,
            "meta_state": meta_state,
        }


class AEFExpert(nn.Module):
    def __init__(
        self,
        mode: str,
        d_model: int,
        hidden_dim: int,
        pred_len: int,
        n_vars: int,
        rank: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        if self.mode == "trend":
            summary_dim = d_model * 2
        elif self.mode == "periodic":
            summary_dim = d_model * 2
        elif self.mode == "boundary":
            summary_dim = d_model * 4
        elif self.mode == "interaction":
            summary_dim = d_model * 2
        else:
            raise ValueError(f"Unsupported expert mode: {mode}")

        self.summary_proj = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.var_proj = nn.Linear(d_model, d_model)
        self.state_norm = nn.LayerNorm(d_model)
        self.head = VariableAwareForecastHead(
            state_dim=d_model,
            token_dim=d_model,
            hidden_dim=hidden_dim,
            out_len=pred_len,
            n_vars=n_vars,
            rank=rank,
            dropout=dropout,
        )

    def _summary(self, context: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == "trend":
            return torch.cat([context["scale_state"], context["temp_tokens"].mean(dim=1)], dim=-1)
        if self.mode == "periodic":
            return torch.cat([context["scale_tokens"].max(dim=1).values, context["scale_state"]], dim=-1)
        if self.mode == "boundary":
            pooled = F.adaptive_avg_pool1d(context["boundary_tokens"].transpose(1, 2), 4)
            return pooled.flatten(start_dim=1)
        if self.mode == "interaction":
            return torch.cat(
                [context["var_tokens"].mean(dim=1), context["var_tokens"].max(dim=1).values],
                dim=-1,
            )
        raise RuntimeError("invalid expert mode")

    def forward(self, context: dict[str, torch.Tensor], fused_state: torch.Tensor) -> torch.Tensor:
        summary = self.summary_proj(self._summary(context))
        state = self.state_norm(fused_state + summary)
        return self.head(state, self.var_proj(context["var_tokens"]))


class AEFPlus(nn.Module):
    def __init__(self, config: AEFPlusConfig) -> None:
        super().__init__()
        self.config = config
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.n_vars = int(config.enc_in)

        self.revin = RevIN(num_features=self.n_vars, affine=True)
        self.backbone = AEFBackbone(
            seq_len=self.seq_len,
            n_vars=self.n_vars,
            d_model=int(config.d_model),
            patch_len=int(config.patch_len),
            patch_stride=int(config.patch_stride),
            n_blocks=int(config.n_blocks),
            n_heads=int(config.n_heads),
            ffn_ratio=int(config.ffn_ratio),
            dropout=float(config.dropout),
            stochastic_depth=float(config.stochastic_depth),
            max_boundary_steps=min(int(config.max_boundary_steps), max(8, self.seq_len // 2)),
            metadata_num_dim=int(config.metadata_num_dim),
            metadata_dim=int(config.metadata_dim),
            artifact_vocab_size=int(config.artifact_vocab_size),
            phase_vocab_size=int(config.phase_vocab_size),
            severity_vocab_size=int(config.severity_vocab_size),
            nvar_vocab_size=int(config.nvar_vocab_size),
            horizon_vocab_size=int(config.horizon_vocab_size),
        )
        self.horizon_route_embed = nn.Embedding(max(int(config.horizon_vocab_size), 1), 16)
        self.phase_route_embed = nn.Embedding(max(int(config.phase_vocab_size), 1), 16)
        router_input_dim = int(config.d_model) * 4 + 16 + 16
        self.router = TopKRouter(
            in_dim=router_input_dim,
            hidden_dim=int(config.expert_hidden),
            num_experts=int(config.num_experts),
            dropout=float(config.dropout),
            top_k=min(2, int(config.num_experts)),
        )
        expert_modes = ["trend", "periodic", "boundary", "interaction"][: int(config.num_experts)]
        self.experts = nn.ModuleList(
            [
                AEFExpert(
                    mode=mode,
                    d_model=int(config.d_model),
                    hidden_dim=int(config.expert_hidden),
                    pred_len=self.pred_len,
                    n_vars=self.n_vars,
                    rank=int(config.head_rank),
                    dropout=float(config.dropout),
                )
                for mode in expert_modes
            ]
        )

        aux_dim = int(config.d_model) * 2
        self.group_head = ClassifierHead(aux_dim, int(config.artifact_vocab_size), float(config.dropout))
        self.phase_head = ClassifierHead(aux_dim, int(config.phase_vocab_size), float(config.dropout))
        self.severity_head = RegressionHead(aux_dim, float(config.dropout))

    def forward(
        self,
        x_raw: torch.Tensor,
        metadata_num: torch.Tensor,
        artifact_id: torch.Tensor,
        phase_id: torch.Tensor,
        severity_bin_id: torch.Tensor,
        nvar_bin_id: torch.Tensor,
        horizon_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x_norm, stats = self.revin.normalize(x_raw)
        context = self.backbone(
            x_norm=x_norm,
            metadata_num=metadata_num,
            artifact_id=artifact_id,
            phase_id=phase_id,
            severity_bin_id=severity_bin_id,
            nvar_bin_id=nvar_bin_id,
            horizon_id=horizon_id,
        )

        router_input = torch.cat(
            [
                context["state"],
                context["meta_state"],
                context["boundary_state"],
                context["scale_state"],
                self.horizon_route_embed(horizon_id),
                self.phase_route_embed(phase_id),
            ],
            dim=-1,
        )
        router_weights = self.router(router_input)
        expert_preds = torch.stack(
            [expert(context, context["state"]) for expert in self.experts],
            dim=1,
        )
        pred_norm = torch.sum(router_weights.unsqueeze(-1).unsqueeze(-1) * expert_preds, dim=1)
        aux_state = torch.cat([context["state"], context["meta_state"]], dim=-1)
        return {
            "pred": self.revin.denormalize(pred_norm, stats),
            "router_weights": router_weights,
            "group_logits": self.group_head(aux_state),
            "phase_logits": self.phase_head(aux_state),
            "severity_pred": self.severity_head(aux_state),
        }

    @torch.no_grad()
    def predict(
        self,
        x_raw: torch.Tensor,
        metadata_num: torch.Tensor,
        artifact_id: torch.Tensor,
        phase_id: torch.Tensor,
        severity_bin_id: torch.Tensor,
        nvar_bin_id: torch.Tensor,
        horizon_id: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(
            x_raw=x_raw,
            metadata_num=metadata_num,
            artifact_id=artifact_id,
            phase_id=phase_id,
            severity_bin_id=severity_bin_id,
            nvar_bin_id=nvar_bin_id,
            horizon_id=horizon_id,
        )["pred"]
