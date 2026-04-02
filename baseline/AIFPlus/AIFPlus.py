from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

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
    TemporalPatchEncoder,
    TopKRouter,
    VariableAwareForecastHead,
    VariateTokenEncoder,
)


@dataclass
class AIFPlusConfig:
    seq_len: int
    pred_len: int
    enc_in: int
    metadata_num_dim: int = 10
    artifact_vocab_size: int = 1
    phase_vocab_size: int = 1
    dataset_vocab_size: int = 1
    support_vocab_size: int = 1
    horizon_vocab_size: int = 1
    d_model: int = 256
    latent_dim: int = 256
    expert_hidden: int = 384
    head_rank: int = 32
    patch_len: int = 8
    patch_stride: int = 4
    n_blocks: int = 4
    n_heads: int = 8
    ffn_ratio: int = 4
    dropout: float = 0.1
    stochastic_depth: float = 0.1
    num_experts: int = 4
    epsilon_nuisance: float = 0.05
    use_diff_branch: bool = False




def _compatible_heads(requested_heads: int, d_model: int) -> int:
    requested_heads = int(max(1, requested_heads))
    d_model = int(max(1, d_model))
    for heads in range(min(requested_heads, d_model), 0, -1):
        if d_model % heads == 0:
            return heads
    return 1


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class LatentMLP(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AIFMetadataEncoder(nn.Module):
    def __init__(
        self,
        metadata_num_dim: int,
        latent_dim: int,
        dataset_vocab_size: int,
        support_vocab_size: int,
        horizon_vocab_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.numeric = nn.Sequential(
            nn.LayerNorm(metadata_num_dim),
            nn.Linear(metadata_num_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, latent_dim // 2),
        )
        self.dataset_embed = nn.Embedding(max(int(dataset_vocab_size), 1), 16)
        self.support_embed = nn.Embedding(max(int(support_vocab_size), 1), 8)
        self.horizon_embed = nn.Embedding(max(int(horizon_vocab_size), 1), 16)
        self.proj = nn.Sequential(
            nn.LayerNorm(latent_dim // 2 + 16 + 8 + 16),
            nn.Linear(latent_dim // 2 + 16 + 8 + 16, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        dataset_id: torch.Tensor,
        support_id: torch.Tensor,
        horizon_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dataset = self.dataset_embed(dataset_id)
        support = self.support_embed(support_id)
        horizon = self.horizon_embed(horizon_id)
        route_feats = torch.cat([dataset, support, horizon], dim=-1)
        state = self.proj(torch.cat([self.numeric(numeric), dataset, support, horizon], dim=-1))
        return state, route_feats


class AIFBackbone(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_vars: int,
        feature_multiplier: int,
        d_model: int,
        patch_len: int,
        patch_stride: int,
        n_blocks: int,
        n_heads: int,
        ffn_ratio: int,
        dropout: float,
        stochastic_depth: float,
    ) -> None:
        super().__init__()
        self.temporal_encoder = TemporalPatchEncoder(
            input_dim=n_vars * feature_multiplier,
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
            per_var_feature_dim=feature_multiplier,
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
        self.type_embedding = nn.Parameter(torch.zeros(3, d_model))
        self.seed_proj = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
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
        branch_input: torch.Tensor,
        per_var_features: torch.Tensor,
        normalized_series: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        temp_tokens, temp_state = self.temporal_encoder(branch_input)
        var_tokens, var_state = self.variate_encoder(per_var_features)
        scale_tokens, scale_state = self.scale_encoder(normalized_series)

        temp_tokens = temp_tokens + self.type_embedding[0].view(1, 1, -1)
        var_tokens = var_tokens + self.type_embedding[1].view(1, 1, -1)
        scale_tokens = scale_tokens + self.type_embedding[2].view(1, 1, -1)

        seed = self.seed_proj(torch.cat([temp_state, var_state, scale_state], dim=-1))
        state = self.fusion(seed=seed, context=torch.cat([temp_tokens, var_tokens, scale_tokens], dim=1))
        return {
            "state": state,
            "temp_tokens": temp_tokens,
            "var_tokens": var_tokens,
            "scale_tokens": scale_tokens,
            "temp_state": temp_state,
            "var_state": var_state,
            "scale_state": scale_state,
        }


class AIFExpert(nn.Module):
    def __init__(
        self,
        mode: str,
        d_model: int,
        latent_dim: int,
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
        elif self.mode == "transition":
            summary_dim = d_model * 4
        elif self.mode == "interaction":
            summary_dim = d_model * 2
        else:
            raise ValueError(f"Unsupported expert mode: {mode}")

        self.summary_proj = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.var_proj = nn.Linear(d_model, d_model)
        self.state_norm = nn.LayerNorm(latent_dim)
        self.head = VariableAwareForecastHead(
            state_dim=latent_dim,
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
        if self.mode == "transition":
            pooled = F.adaptive_avg_pool1d(context["temp_tokens"].transpose(1, 2), 4)
            return pooled.flatten(start_dim=1)
        if self.mode == "interaction":
            return torch.cat(
                [context["var_tokens"].mean(dim=1), context["var_tokens"].max(dim=1).values],
                dim=-1,
            )
        raise RuntimeError("invalid expert mode")

    def forward(self, context: dict[str, torch.Tensor], decode_state: torch.Tensor) -> torch.Tensor:
        summary = self.summary_proj(self._summary(context))
        state = self.state_norm(decode_state + summary)
        var_tokens = self.var_proj(context["var_tokens"])
        return self.head(state, var_tokens)


class AIFPlus(nn.Module):
    def __init__(self, config: AIFPlusConfig) -> None:
        super().__init__()
        self.config = config
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.n_vars = int(config.enc_in)
        self.use_diff_branch = bool(config.use_diff_branch)
        self.epsilon_nuisance = float(config.epsilon_nuisance)
        self.feature_multiplier = 3 if self.use_diff_branch else 2

        self.revin = RevIN(num_features=self.n_vars, affine=True)
        self.uncertainty_proj = nn.Parameter(torch.tensor(0.2))

        self.backbone = AIFBackbone(
            seq_len=self.seq_len,
            n_vars=self.n_vars,
            feature_multiplier=self.feature_multiplier,
            d_model=int(config.d_model),
            patch_len=int(config.patch_len),
            patch_stride=int(config.patch_stride),
            n_blocks=int(config.n_blocks),
            n_heads=int(config.n_heads),
            ffn_ratio=int(config.ffn_ratio),
            dropout=float(config.dropout),
            stochastic_depth=float(config.stochastic_depth),
        )
        self.clean_branch = LatentMLP(in_dim=int(config.d_model), latent_dim=int(config.latent_dim), dropout=float(config.dropout))
        self.art_branch = LatentMLP(in_dim=int(config.d_model), latent_dim=int(config.latent_dim), dropout=float(config.dropout))
        self.art_delta_proj = nn.Sequential(
            nn.LayerNorm(int(config.d_model)),
            nn.Linear(int(config.d_model), int(config.d_model)),
            nn.GELU(),
        )
        self.meta_encoder = AIFMetadataEncoder(
            metadata_num_dim=int(config.metadata_num_dim),
            latent_dim=int(config.latent_dim),
            dataset_vocab_size=int(config.dataset_vocab_size),
            support_vocab_size=int(config.support_vocab_size),
            horizon_vocab_size=int(config.horizon_vocab_size),
            dropout=float(config.dropout),
        )
        self.clean_fusion = nn.Sequential(
            nn.LayerNorm(int(config.latent_dim) * 2),
            nn.Linear(int(config.latent_dim) * 2, int(config.latent_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
        )
        self.nuisance_gate = nn.Sequential(
            nn.LayerNorm(int(config.latent_dim) * 3),
            nn.Linear(int(config.latent_dim) * 3, int(config.latent_dim)),
            nn.Sigmoid(),
        )
        self.nuisance_bridge = nn.Sequential(
            nn.LayerNorm(int(config.latent_dim)),
            nn.Linear(int(config.latent_dim), int(config.latent_dim)),
            nn.GELU(),
        )
        router_input_dim = int(config.latent_dim) * 2 + int(config.d_model) * 2 + (16 + 8 + 16)
        self.router = TopKRouter(
            in_dim=router_input_dim,
            hidden_dim=int(config.expert_hidden),
            num_experts=int(config.num_experts),
            dropout=float(config.dropout),
            top_k=min(2, int(config.num_experts)),
        )
        expert_modes = ["trend", "periodic", "transition", "interaction"][: int(config.num_experts)]
        self.experts = nn.ModuleList(
            [
                AIFExpert(
                    mode=mode,
                    d_model=int(config.d_model),
                    latent_dim=int(config.latent_dim),
                    hidden_dim=int(config.expert_hidden),
                    pred_len=self.pred_len,
                    n_vars=self.n_vars,
                    rank=int(config.head_rank),
                    dropout=float(config.dropout),
                )
                for mode in expert_modes
            ]
        )
        self.reconstruction_head = VariableAwareForecastHead(
            state_dim=int(config.latent_dim),
            token_dim=int(config.d_model),
            hidden_dim=int(config.expert_hidden),
            out_len=self.seq_len,
            n_vars=self.n_vars,
            rank=max(16, int(config.head_rank) // 2),
            dropout=float(config.dropout),
        )

        self.grl = GradientReversal(alpha=1.0)
        self.artifact_adv_head = ClassifierHead(int(config.latent_dim), int(config.artifact_vocab_size), float(config.dropout))
        self.phase_adv_head = ClassifierHead(int(config.latent_dim), int(config.phase_vocab_size), float(config.dropout))
        self.artifact_aux_head = ClassifierHead(int(config.latent_dim), int(config.artifact_vocab_size), float(config.dropout))
        self.phase_aux_head = ClassifierHead(int(config.latent_dim), int(config.phase_vocab_size), float(config.dropout))

    @staticmethod
    def _first_difference(x: torch.Tensor) -> torch.Tensor:
        diff = x[:, 1:, :] - x[:, :-1, :]
        pad = torch.zeros_like(x[:, :1, :])
        return torch.cat([pad, diff], dim=1)

    def _scaled_uncertainty(self, uncertainty: torch.Tensor) -> torch.Tensor:
        return uncertainty * torch.tanh(self.uncertainty_proj)

    def _build_branch_input(self, x: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        uncertainty_scaled = self._scaled_uncertainty(uncertainty)
        features = [x, uncertainty_scaled]
        if self.use_diff_branch:
            features.append(self._first_difference(x))
        return torch.cat(features, dim=-1)

    def _build_per_var_features(self, x: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        uncertainty_scaled = self._scaled_uncertainty(uncertainty)
        features = [x.unsqueeze(-1), uncertainty_scaled.unsqueeze(-1)]
        if self.use_diff_branch:
            features.append(self._first_difference(x).unsqueeze(-1))
        return torch.cat(features, dim=-1)

    def forward(
        self,
        x_raw: torch.Tensor,
        x_masked: torch.Tensor,
        uncertainty: torch.Tensor,
        metadata_num: torch.Tensor,
        dataset_id: torch.Tensor,
        support_id: torch.Tensor,
        horizon_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raw_norm, _ = self.revin.normalize(x_raw)
        masked_norm, masked_stats = self.revin.normalize(x_masked)

        raw_context = self.backbone(
            branch_input=self._build_branch_input(raw_norm, uncertainty),
            per_var_features=self._build_per_var_features(raw_norm, uncertainty),
            normalized_series=raw_norm,
        )
        masked_context = self.backbone(
            branch_input=self._build_branch_input(masked_norm, uncertainty),
            per_var_features=self._build_per_var_features(masked_norm, uncertainty),
            normalized_series=masked_norm,
        )

        z_clean = self.clean_branch(masked_context["state"])
        artifact_seed = raw_context["state"] + self.art_delta_proj(raw_context["state"] - masked_context["state"])
        z_art = self.art_branch(artifact_seed)

        meta_vec, route_meta = self.meta_encoder(
            numeric=metadata_num,
            dataset_id=dataset_id,
            support_id=support_id,
            horizon_id=horizon_id,
        )
        clean_state = self.clean_fusion(torch.cat([z_clean, meta_vec], dim=-1))
        nuisance_gate = self.nuisance_gate(torch.cat([z_clean, z_art, meta_vec], dim=-1))
        nuisance_hint = self.nuisance_bridge(z_art) * nuisance_gate * self.epsilon_nuisance
        decode_state = clean_state + nuisance_hint

        router_input = torch.cat(
            [
                decode_state,
                meta_vec,
                masked_context["scale_state"],
                masked_context["var_state"],
                route_meta,
            ],
            dim=-1,
        )
        router_weights = self.router(router_input)

        expert_preds = torch.stack(
            [expert(masked_context, decode_state) for expert in self.experts],
            dim=1,
        )
        pred_norm = torch.sum(router_weights.unsqueeze(-1).unsqueeze(-1) * expert_preds, dim=1)
        rec_norm = self.reconstruction_head(clean_state, masked_context["var_tokens"])

        return {
            "pred": self.revin.denormalize(pred_norm, masked_stats),
            "reconstruction": self.revin.denormalize(rec_norm, masked_stats),
            "router_weights": router_weights,
            "z_clean": z_clean,
            "z_art": z_art,
            "artifact_logits_clean": self.artifact_adv_head(self.grl(z_clean)),
            "phase_logits_clean": self.phase_adv_head(self.grl(z_clean)),
            "artifact_logits_art": self.artifact_aux_head(z_art),
            "phase_logits_art": self.phase_aux_head(z_art),
        }

    @torch.no_grad()
    def predict(
        self,
        x_raw: torch.Tensor,
        x_masked: torch.Tensor,
        uncertainty: torch.Tensor,
        metadata_num: torch.Tensor,
        dataset_id: torch.Tensor,
        support_id: torch.Tensor,
        horizon_id: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(
            x_raw=x_raw,
            x_masked=x_masked,
            uncertainty=uncertainty,
            metadata_num=metadata_num,
            dataset_id=dataset_id,
            support_id=support_id,
            horizon_id=horizon_id,
        )["pred"]
