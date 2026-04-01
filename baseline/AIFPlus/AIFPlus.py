from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


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


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mean = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        x_norm = (x - mean) / stdev
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm, (mean, stdev)

    def denormalize(self, x: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mean, stdev = stats
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        return x * stdev + mean


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)
        return x * random_tensor / keep_prob


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiKernelTemporalConv(nn.Module):
    def __init__(self, d_model: int, dropout: float, kernels: tuple[int, ...]) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=d_model,
                )
                for kernel_size in kernels
            ]
        )
        self.out_proj = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        mixed = torch.stack([conv(x_t) for conv in self.convs], dim=0).mean(dim=0)
        return self.out_proj(mixed).transpose(1, 2)


class ResolutionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_ratio: int,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_conv = nn.LayerNorm(d_model)
        self.temporal_conv = MultiKernelTemporalConv(d_model=d_model, dropout=dropout, kernels=(3, 5, 9, 17))
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, hidden_dim=d_model * ffn_ratio, dropout=dropout)
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm_attn(x)
        attn_out, _ = self.self_attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.temporal_conv(self.norm_conv(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x


class SharedTrunk(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int,
        patch_len: int,
        patch_stride: int,
        n_blocks: int,
        n_resolutions: int,
        n_heads: int,
        ffn_ratio: int,
        dropout: float,
        stochastic_depth: float,
    ) -> None:
        super().__init__()
        self.n_resolutions = int(n_resolutions)
        self.patch_embed = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=patch_stride,
            padding=patch_len // 2,
        )
        with torch.no_grad():
            dummy = torch.zeros(1, input_dim, seq_len)
            token_len = int(self.patch_embed(dummy).shape[-1])
        self.positional = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, max(1, math.ceil(token_len / (2**idx))), d_model))
                for idx in range(self.n_resolutions)
            ]
        )
        self.blocks = nn.ModuleList()
        drop_rates = torch.linspace(0.0, stochastic_depth, steps=max(n_blocks, 1)).tolist()
        for block_idx in range(n_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        ResolutionBlock(
                            d_model=d_model,
                            n_heads=n_heads,
                            ffn_ratio=ffn_ratio,
                            dropout=dropout,
                            drop_path=drop_rates[block_idx],
                        )
                        for _ in range(self.n_resolutions)
                    ]
                )
            )
        self.fusion_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(self.n_resolutions)])
        self.summary = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _resize(tokens: torch.Tensor, target_length: int) -> torch.Tensor:
        if int(tokens.shape[1]) == int(target_length):
            return tokens
        return F.interpolate(
            tokens.transpose(1, 2),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def _build_resolutions(self, base_tokens: torch.Tensor) -> list[torch.Tensor]:
        outputs = [base_tokens]
        for res_idx in range(1, self.n_resolutions):
            pooled = F.avg_pool1d(
                base_tokens.transpose(1, 2),
                kernel_size=2**res_idx,
                stride=2**res_idx,
                ceil_mode=True,
            ).transpose(1, 2)
            outputs.append(pooled)
        return outputs

    def _fuse_resolutions(self, tokens_list: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for target_idx, target_tokens in enumerate(tokens_list):
            length = int(target_tokens.shape[1])
            aligned = torch.stack([self._resize(tokens, length) for tokens in tokens_list], dim=0).mean(dim=0)
            outputs.append(target_tokens + self.fusion_proj[target_idx](aligned))
        return outputs

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_tokens = self.patch_embed(x.transpose(1, 2)).transpose(1, 2)
        tokens_list = self._build_resolutions(base_tokens)
        tokens_list = [
            tokens + self.positional[idx][:, : tokens.shape[1], :]
            for idx, tokens in enumerate(tokens_list)
        ]
        for block_group in self.blocks:
            tokens_list = [block(tokens) for block, tokens in zip(block_group, tokens_list)]
            tokens_list = self._fuse_resolutions(tokens_list)
        fused_tokens = torch.stack(
            [self._resize(tokens, int(tokens_list[0].shape[1])) for tokens in tokens_list],
            dim=0,
        ).mean(dim=0)
        pooled = torch.cat(
            [
                fused_tokens.mean(dim=1),
                fused_tokens.max(dim=1).values,
                fused_tokens[:, -1, :],
            ],
            dim=-1,
        )
        return fused_tokens, self.summary(pooled)


class LatentMLP(nn.Module):
    def __init__(self, d_model: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetadataEncoder(nn.Module):
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
        self.dataset_embed = nn.Embedding(max(dataset_vocab_size, 1), 16)
        self.support_embed = nn.Embedding(max(support_vocab_size, 1), 8)
        self.horizon_embed = nn.Embedding(max(horizon_vocab_size, 1), 16)
        self.proj = nn.Sequential(
            nn.Linear(latent_dim // 2 + 16 + 8 + 16, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        dataset_id: torch.Tensor,
        support_id: torch.Tensor,
        horizon_id: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                self.numeric(numeric),
                self.dataset_embed(dataset_id),
                self.support_embed(support_id),
                self.horizon_embed(horizon_id),
            ],
            dim=-1,
        )
        return self.proj(features)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.num_classes = int(max(num_classes, 1))
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FactorizedForecastHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_len: int, n_vars: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.out_len = int(out_len)
        self.n_vars = int(n_vars)
        self.rank = int(rank)
        self.time_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_len * rank),
        )
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_vars * rank),
        )
        self.bias = nn.Parameter(torch.zeros(out_len, n_vars))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_basis = self.time_mlp(x).view(x.shape[0], self.out_len, self.rank)
        channel_basis = self.channel_mlp(x).view(x.shape[0], self.n_vars, self.rank)
        output = torch.einsum("blr,bcr->blc", time_basis, channel_basis) / math.sqrt(float(self.rank))
        return output + self.bias


class ExpertModule(nn.Module):
    def __init__(self, mode: str, d_model: int, latent_dim: int, hidden_dim: int, pred_len: int, n_vars: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.mode = str(mode)
        if self.mode == "trend":
            self.temporal = nn.AdaptiveAvgPool1d(4)
            summary_dim = d_model * 4
        elif self.mode == "periodic":
            self.temporal = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
            summary_dim = d_model
        elif self.mode == "transition":
            self.temporal = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2, groups=d_model)
            summary_dim = d_model * 4
        elif self.mode == "interaction":
            self.temporal = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),
            )
            summary_dim = d_model
        else:
            raise ValueError(f"Unsupported expert mode: {mode}")

        self.summary_proj = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = FactorizedForecastHead(
            in_dim=latent_dim * 2,
            hidden_dim=hidden_dim,
            out_len=pred_len,
            n_vars=n_vars,
            rank=rank,
            dropout=dropout,
        )

    def _summarize(self, tokens: torch.Tensor) -> torch.Tensor:
        x_t = tokens.transpose(1, 2)
        if self.mode == "trend":
            return self.temporal(x_t).flatten(start_dim=1)
        if self.mode == "periodic":
            return self.temporal(x_t).mean(dim=-1)
        if self.mode == "transition":
            return self.temporal(x_t)[:, :, -4:].flatten(start_dim=1)
        if self.mode == "interaction":
            mixed = self.temporal(x_t)
            return mixed.mean(dim=-1) + mixed.max(dim=-1).values
        raise RuntimeError("invalid expert mode")

    def forward(self, tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        summary = self.summary_proj(self._summarize(tokens))
        return self.head(torch.cat([state, summary], dim=-1))


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

        self.shared_trunk = SharedTrunk(
            input_dim=self.n_vars * self.feature_multiplier,
            seq_len=self.seq_len,
            d_model=int(config.d_model),
            patch_len=int(config.patch_len),
            patch_stride=int(config.patch_stride),
            n_blocks=int(config.n_blocks),
            n_resolutions=3,
            n_heads=int(config.n_heads),
            ffn_ratio=int(config.ffn_ratio),
            dropout=float(config.dropout),
            stochastic_depth=float(config.stochastic_depth),
        )
        self.clean_branch = LatentMLP(d_model=int(config.d_model), latent_dim=int(config.latent_dim), dropout=float(config.dropout))
        self.art_branch = LatentMLP(d_model=int(config.d_model), latent_dim=int(config.latent_dim), dropout=float(config.dropout))
        self.meta_encoder = MetadataEncoder(
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
        self.nuisance_bridge = nn.Sequential(
            nn.LayerNorm(int(config.latent_dim)),
            nn.Linear(int(config.latent_dim), int(config.latent_dim)),
            nn.GELU(),
        )
        router_input_dim = int(config.latent_dim) * 2 + 16 + 8 + 16
        self.router = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, int(config.expert_hidden)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.expert_hidden), int(config.num_experts)),
        )
        expert_modes = ["trend", "periodic", "transition", "interaction"][: int(config.num_experts)]
        self.experts = nn.ModuleList(
            [
                ExpertModule(
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
        self.reconstruction_head = FactorizedForecastHead(
            in_dim=int(config.latent_dim),
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

    def _build_branch_input(self, x: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        uncertainty_scaled = uncertainty * torch.tanh(self.uncertainty_proj)
        features = [x, uncertainty_scaled]
        if self.use_diff_branch:
            features.append(self._first_difference(x))
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
        raw_tokens, raw_pooled = self.shared_trunk(self._build_branch_input(raw_norm, uncertainty))
        masked_tokens, masked_pooled = self.shared_trunk(self._build_branch_input(masked_norm, uncertainty))

        z_clean = self.clean_branch(masked_pooled)
        z_art = self.art_branch(raw_pooled)
        meta_vec = self.meta_encoder(
            numeric=metadata_num,
            dataset_id=dataset_id,
            support_id=support_id,
            horizon_id=horizon_id,
        )
        clean_state = self.clean_fusion(torch.cat([z_clean, meta_vec], dim=-1))
        nuisance_hint = self.nuisance_bridge(z_art).detach() * self.epsilon_nuisance
        decode_state = clean_state + nuisance_hint
        router_input = torch.cat(
            [
                decode_state,
                meta_vec,
                self.meta_encoder.dataset_embed(dataset_id),
                self.meta_encoder.support_embed(support_id),
                self.meta_encoder.horizon_embed(horizon_id),
            ],
            dim=-1,
        )
        router_weights = torch.softmax(self.router(router_input), dim=-1)

        expert_preds = torch.stack([expert(masked_tokens, decode_state) for expert in self.experts], dim=1)
        pred_norm = torch.sum(router_weights.unsqueeze(-1).unsqueeze(-1) * expert_preds, dim=1)
        rec_norm = self.reconstruction_head(clean_state)

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
