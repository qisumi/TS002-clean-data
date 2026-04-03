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
    FeedForward,
    RevIN,
    ScaleFrequencyEncoder,
    StochasticDepth,
    TemporalMixerBlock,
    VariableAwareForecastHead,
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
    patch_len: int = 16
    patch_stride: int = 8
    n_blocks: int = 3
    n_heads: int = 8
    ffn_ratio: int = 4
    dropout: float = 0.1
    stochastic_depth: float = 0.05
    num_experts: int = 2
    epsilon_nuisance: float = 0.05
    use_diff_branch: bool = False
    num_queries: int = 32
    num_tq_layers: int = 1
    query_period: int = 24
    resid_hidden: int = 64
    lambda_res_init: float = 0.01
    lambda_res_max: float = 0.05
    scales: tuple[int, ...] = (1, 2, 4)


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


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_ratio: int,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_ctx = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model * ffn_ratio, dropout)
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        ctx = self.norm_ctx(context)
        out, _ = self.cross(q, ctx, ctx, need_weights=False)
        query = query + self.drop_path(out)
        query = query + self.drop_path(self.ffn(self.norm_ffn(query)))
        return query


class ChannelIndependentPatchEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        d_model: int,
        patch_len: int,
        patch_stride: int,
        n_layers: int,
        n_heads: int,
        ffn_ratio: int,
        dropout: float,
        stochastic_depth: float,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.patch_embed = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.d_model,
            kernel_size=int(patch_len),
            stride=int(patch_stride),
            padding=int(patch_len) // 2,
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.feature_dim, self.seq_len)
            token_len = int(self.patch_embed(dummy).shape[-1])
        self.positional = nn.Parameter(torch.zeros(1, token_len, self.d_model))
        self.layers = nn.ModuleList()
        drop_rates = torch.linspace(0.0, float(stochastic_depth), steps=max(int(n_layers), 1)).tolist()
        attn_heads = _compatible_heads(n_heads, self.d_model)
        for idx in range(max(int(n_layers), 1)):
            self.layers.append(
                AttentionBlock(
                    d_model=self.d_model,
                    n_heads=attn_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    drop_path=drop_rates[idx],
                )
            )
        self.var_summary = nn.Sequential(
            nn.LayerNorm(self.d_model * 3),
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_summary = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, per_var_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if per_var_features.ndim != 4:
            raise ValueError(f"Expected 4D input [B, L, C, F], got {tuple(per_var_features.shape)}")
        batch_size, seq_len, n_vars, feature_dim = per_var_features.shape
        if seq_len != self.seq_len or feature_dim != self.feature_dim:
            raise ValueError(
                "Unexpected per-variable feature shape. "
                f"Expected [B, {self.seq_len}, C, {self.feature_dim}], "
                f"got [B, {seq_len}, {n_vars}, {feature_dim}]"
            )

        x = per_var_features.permute(0, 2, 3, 1).reshape(batch_size * n_vars, self.feature_dim, self.seq_len)
        patch_tokens = self.patch_embed(x).transpose(1, 2)
        patch_tokens = patch_tokens + self.positional[:, : patch_tokens.shape[1], :]
        for layer in self.layers:
            patch_tokens = layer(patch_tokens)

        pooled = torch.cat(
            [
                patch_tokens.mean(dim=1),
                patch_tokens.max(dim=1).values,
                patch_tokens[:, -1, :],
            ],
            dim=-1,
        )
        var_tokens = self.var_summary(pooled).view(batch_size, n_vars, self.d_model)
        global_state = self.global_summary(
            torch.cat([var_tokens.mean(dim=1), var_tokens.max(dim=1).values], dim=-1)
        )
        return {
            "patch_tokens": patch_tokens.view(batch_size, n_vars, patch_tokens.shape[1], self.d_model),
            "var_tokens": var_tokens,
            "state": global_state,
        }


class PeriodicQueryCrossBranch(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_queries: int,
        num_layers: int,
        query_period: int,
        ffn_ratio: int,
        dropout: float,
        stochastic_depth: float,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_queries = int(max(1, num_queries))
        self.query_period = int(max(1, query_period))
        self.query_bank = nn.Parameter(torch.zeros(1, self.num_queries, self.d_model))
        self.period_embed = nn.Embedding(self.query_period, self.d_model)
        self.layers = nn.ModuleList()
        drop_rates = torch.linspace(0.0, float(stochastic_depth), steps=max(int(num_layers), 1)).tolist()
        attn_heads = _compatible_heads(n_heads, self.d_model)
        for idx in range(max(int(num_layers), 1)):
            self.layers.append(
                CrossAttentionBlock(
                    d_model=self.d_model,
                    n_heads=attn_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    drop_path=drop_rates[idx],
                )
            )
        self.summary = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.var_fusion = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _build_queries(self, batch_size: int, device: torch.device) -> torch.Tensor:
        query_index = torch.arange(self.num_queries, device=device)
        periodic_bias = self.period_embed(query_index % self.query_period)
        return self.query_bank.expand(batch_size, -1, -1) + periodic_bias.unsqueeze(0)

    def forward(self, var_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = int(var_tokens.shape[0])
        queries = self._build_queries(batch_size=batch_size, device=var_tokens.device)
        for layer in self.layers:
            queries = layer(queries, var_tokens)
        state = self.summary(torch.cat([queries.mean(dim=1), queries.max(dim=1).values], dim=-1))
        state_broadcast = state.unsqueeze(1).expand(-1, var_tokens.shape[1], -1)
        refined_var_tokens = var_tokens + self.var_fusion(torch.cat([var_tokens, state_broadcast], dim=-1))
        return {
            "query_tokens": queries,
            "var_tokens": refined_var_tokens,
            "state": state,
        }


class ArtifactResidualBranch(nn.Module):
    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        hidden_dim: int,
        pred_len: int,
        n_vars: int,
        rank: int,
        head_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_vars = int(n_vars)
        self.input_proj = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.hidden_dim,
            kernel_size=5,
            padding=2,
        )
        self.mix_blocks = nn.ModuleList(
            [
                TemporalMixerBlock(
                    d_model=self.hidden_dim,
                    ffn_ratio=2,
                    dropout=dropout,
                    drop_path=0.0,
                    kernel_size=5,
                    dilation=1,
                ),
                TemporalMixerBlock(
                    d_model=self.hidden_dim,
                    ffn_ratio=2,
                    dropout=dropout,
                    drop_path=0.0,
                    kernel_size=7,
                    dilation=2,
                ),
            ]
        )
        self.var_summary = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.state_summary = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = VariableAwareForecastHead(
            state_dim=self.hidden_dim,
            token_dim=self.hidden_dim,
            hidden_dim=int(max(head_hidden // 2, self.hidden_dim * 2)),
            out_len=int(pred_len),
            n_vars=self.n_vars,
            rank=max(8, int(rank) // 2),
            dropout=dropout,
        )

    def forward(self, per_var_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if per_var_features.ndim != 4:
            raise ValueError(f"Expected 4D input [B, L, C, F], got {tuple(per_var_features.shape)}")
        batch_size, seq_len, n_vars, feature_dim = per_var_features.shape
        if seq_len != self.seq_len or n_vars != self.n_vars or feature_dim != self.feature_dim:
            raise ValueError(
                "Unexpected residual feature shape. "
                f"Expected [B, {self.seq_len}, {self.n_vars}, {self.feature_dim}], "
                f"got [B, {seq_len}, {n_vars}, {feature_dim}]"
            )
        x = per_var_features.permute(0, 2, 3, 1).reshape(batch_size * self.n_vars, self.feature_dim, self.seq_len)
        tokens = self.input_proj(x).transpose(1, 2)
        for block in self.mix_blocks:
            tokens = block(tokens)
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)
        var_tokens = self.var_summary(pooled).view(batch_size, self.n_vars, self.hidden_dim)
        state = self.state_summary(torch.cat([var_tokens.mean(dim=1), var_tokens.max(dim=1).values], dim=-1))
        return {
            "pred": self.head(state, var_tokens),
            "var_tokens": var_tokens,
            "state": state,
        }


class AIFPlus(nn.Module):
    def __init__(self, config: AIFPlusConfig) -> None:
        super().__init__()
        self.config = config
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.n_vars = int(config.enc_in)
        self.d_model = int(config.d_model)
        self.latent_dim = int(config.latent_dim)
        self.use_diff_branch = bool(config.use_diff_branch)
        self.clean_feature_dim = 3 if self.use_diff_branch else 2
        self.residual_feature_dim = 3 if self.use_diff_branch else 2
        self.lambda_res_max = float(max(config.lambda_res_max, 0.0))

        self.revin = RevIN(num_features=self.n_vars, affine=True)
        self.uncertainty_proj = nn.Parameter(torch.tensor(0.2))

        self.clean_patch = ChannelIndependentPatchEncoder(
            seq_len=self.seq_len,
            feature_dim=self.clean_feature_dim,
            d_model=self.d_model,
            patch_len=int(config.patch_len),
            patch_stride=int(config.patch_stride),
            n_layers=max(2, int(config.n_blocks)),
            n_heads=int(config.n_heads),
            ffn_ratio=int(config.ffn_ratio),
            dropout=float(config.dropout),
            stochastic_depth=float(config.stochastic_depth),
        )
        self.cross_var_branch = PeriodicQueryCrossBranch(
            d_model=self.d_model,
            n_heads=int(config.n_heads),
            num_queries=int(config.num_queries),
            num_layers=max(1, int(config.num_tq_layers)),
            query_period=int(config.query_period),
            ffn_ratio=int(config.ffn_ratio),
            dropout=float(config.dropout),
            stochastic_depth=float(config.stochastic_depth) * 0.5,
        )
        self.scale_encoder = ScaleFrequencyEncoder(
            n_vars=self.n_vars,
            seq_len=self.seq_len,
            d_model=self.d_model,
            dropout=float(config.dropout),
            scales=tuple(int(scale) for scale in config.scales),
            spectral_topk=8,
        )
        self.horizon_embed = nn.Embedding(max(int(config.horizon_vocab_size), 1), 16)
        self.clean_fusion = nn.Sequential(
            nn.LayerNorm(self.d_model * 3 + 16),
            nn.Linear(self.d_model * 3 + 16, self.latent_dim),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.local_head = VariableAwareForecastHead(
            state_dim=self.latent_dim,
            token_dim=self.d_model,
            hidden_dim=int(config.expert_hidden),
            out_len=self.pred_len,
            n_vars=self.n_vars,
            rank=int(config.head_rank),
            dropout=float(config.dropout),
        )
        self.global_head = VariableAwareForecastHead(
            state_dim=self.latent_dim,
            token_dim=self.d_model,
            hidden_dim=int(config.expert_hidden),
            out_len=self.pred_len,
            n_vars=self.n_vars,
            rank=int(config.head_rank),
            dropout=float(config.dropout),
        )
        self.reconstruction_head = VariableAwareForecastHead(
            state_dim=self.latent_dim,
            token_dim=self.d_model,
            hidden_dim=int(max(config.expert_hidden // 2, config.latent_dim)),
            out_len=self.seq_len,
            n_vars=self.n_vars,
            rank=max(16, int(config.head_rank) // 2),
            dropout=float(config.dropout),
        )
        self.artifact_branch = ArtifactResidualBranch(
            seq_len=self.seq_len,
            feature_dim=self.residual_feature_dim,
            hidden_dim=int(config.resid_hidden),
            pred_len=self.pred_len,
            n_vars=self.n_vars,
            rank=int(config.head_rank),
            head_hidden=int(config.expert_hidden),
            dropout=float(config.dropout),
        )
        self.art_state_proj = nn.Sequential(
            nn.LayerNorm(int(config.resid_hidden)),
            nn.Linear(int(config.resid_hidden), self.latent_dim),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        if self.lambda_res_max > 0.0:
            ratio = min(max(float(config.lambda_res_init) / self.lambda_res_max, 1e-4), 1.0 - 1e-4)
            self.lambda_res_logit = nn.Parameter(torch.logit(torch.tensor(ratio, dtype=torch.float32)))
        else:
            self.lambda_res_logit = nn.Parameter(torch.tensor(-12.0))

        self.grl = GradientReversal(alpha=1.0)
        self.artifact_adv_head = ClassifierHead(self.latent_dim, int(config.artifact_vocab_size), float(config.dropout))
        self.phase_adv_head = ClassifierHead(self.latent_dim, int(config.phase_vocab_size), float(config.dropout))
        self.artifact_aux_head = ClassifierHead(self.latent_dim, int(config.artifact_vocab_size), float(config.dropout))
        self.phase_aux_head = ClassifierHead(self.latent_dim, int(config.phase_vocab_size), float(config.dropout))

    @staticmethod
    def _first_difference(x: torch.Tensor) -> torch.Tensor:
        diff = x[:, 1:, :] - x[:, :-1, :]
        pad = torch.zeros_like(x[:, :1, :])
        return torch.cat([pad, diff], dim=1)

    def _scaled_uncertainty(self, uncertainty: torch.Tensor) -> torch.Tensor:
        return uncertainty * torch.tanh(self.uncertainty_proj)

    def _build_clean_features(self, x: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        uncertainty_scaled = self._scaled_uncertainty(uncertainty)
        features = [x.unsqueeze(-1), uncertainty_scaled.unsqueeze(-1)]
        if self.use_diff_branch:
            features.append(self._first_difference(x).unsqueeze(-1))
        return torch.cat(features, dim=-1)

    def _build_residual_features(self, delta: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        uncertainty_scaled = self._scaled_uncertainty(uncertainty)
        features = [delta.unsqueeze(-1), uncertainty_scaled.unsqueeze(-1)]
        if self.use_diff_branch:
            features.append(self._first_difference(delta).unsqueeze(-1))
        return torch.cat(features, dim=-1)

    def _residual_scale(self) -> torch.Tensor:
        if self.lambda_res_max <= 0.0:
            return self.lambda_res_logit.new_tensor(0.0)
        return torch.sigmoid(self.lambda_res_logit) * self.lambda_res_max

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
        del metadata_num, dataset_id, support_id

        masked_norm, masked_stats = self.revin.normalize(x_masked)
        clean_features = self._build_clean_features(masked_norm, uncertainty)
        patch_context = self.clean_patch(clean_features)
        tq_context = self.cross_var_branch(patch_context["var_tokens"])
        scale_tokens, scale_state = self.scale_encoder(masked_norm)

        horizon_state = self.horizon_embed(horizon_id)
        clean_state = self.clean_fusion(
            torch.cat(
                [
                    patch_context["state"],
                    tq_context["state"],
                    scale_state,
                    horizon_state,
                ],
                dim=-1,
            )
        )

        pred_clean_norm = self.local_head(clean_state, patch_context["var_tokens"]) + self.global_head(
            clean_state,
            tq_context["var_tokens"],
        )
        rec_norm = self.reconstruction_head(clean_state, patch_context["var_tokens"])

        delta = x_raw - x_masked
        art_context = self.artifact_branch(self._build_residual_features(delta, uncertainty))
        residual_scale = self._residual_scale()
        pred_norm = pred_clean_norm + residual_scale * art_context["pred"]
        z_art = self.art_state_proj(art_context["state"])

        return {
            "pred": self.revin.denormalize(pred_norm, masked_stats),
            "reconstruction": self.revin.denormalize(rec_norm, masked_stats),
            "lambda_res": residual_scale.detach(),
            "z_clean": clean_state,
            "z_art": z_art,
            "scale_tokens": scale_tokens,
            "artifact_logits_clean": self.artifact_adv_head(self.grl(clean_state)),
            "phase_logits_clean": self.phase_adv_head(self.grl(clean_state)),
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
