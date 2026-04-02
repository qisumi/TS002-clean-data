
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization with a corrected affine inverse.
    """

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
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        return x * stdev + mean


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.drop_prob <= 0.0:
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


class AttentionBlock(nn.Module):
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
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model * ffn_ratio, dropout)
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm_attn(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x


class LargeKernelMix(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        kernel_size = int(max(3, kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = ((kernel_size - 1) // 2) * dilation
        self.dw = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=d_model,
        )
        self.pw = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        mixed = self.dw(x_t)
        gate, value = self.pw(F.gelu(mixed)).chunk(2, dim=1)
        mixed = torch.sigmoid(gate) * value
        return self.dropout(mixed.transpose(1, 2))


class TemporalMixerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_ratio: int,
        dropout: float,
        drop_path: float,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.norm_mix = nn.LayerNorm(d_model)
        self.mix = LargeKernelMix(d_model, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model * ffn_ratio, dropout)
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.mix(self.norm_mix(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x


class GlobalCrossBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, drop_path: float) -> None:
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
        self.ffn = FeedForward(d_model, d_model * 4, dropout)
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        k = self.norm_ctx(context)
        out, _ = self.cross(q, k, k, need_weights=False)
        query = query + self.drop_path(out)
        query = query + self.drop_path(self.ffn(self.norm_ffn(query)))
        return query


class GlobalFusion(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        stochastic_depth: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        drop_rates = torch.linspace(0.0, float(stochastic_depth), steps=max(int(n_layers), 1)).tolist()
        for idx in range(max(int(n_layers), 1)):
            self.layers.append(
                GlobalCrossBlock(d_model=d_model, n_heads=n_heads, dropout=dropout, drop_path=drop_rates[idx])
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, seed: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = seed.unsqueeze(1)
        for layer in self.layers:
            query = layer(query, context)
        return self.norm(query.squeeze(1))


class TemporalPatchEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int,
        patch_len: int,
        patch_stride: int,
        n_layers: int,
        ffn_ratio: int,
        dropout: float,
        stochastic_depth: float,
    ) -> None:
        super().__init__()
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
        self.positional = nn.Parameter(torch.zeros(1, token_len, d_model))
        self.layers = nn.ModuleList()
        drop_rates = torch.linspace(0.0, float(stochastic_depth), steps=max(int(n_layers), 1)).tolist()
        kernels = [max(5, patch_len + 1), max(7, patch_len + 5), max(9, patch_len + 9)]
        dilations = [1, 2, 1]
        for idx in range(max(int(n_layers), 1)):
            self.layers.append(
                TemporalMixerBlock(
                    d_model=d_model,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    drop_path=drop_rates[idx],
                    kernel_size=kernels[idx % len(kernels)],
                    dilation=dilations[idx % len(dilations)],
                )
            )
        self.summary = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.patch_embed(x.transpose(1, 2)).transpose(1, 2)
        tokens = tokens + self.positional[:, : tokens.shape[1], :]
        for layer in self.layers:
            tokens = layer(tokens)
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values, tokens[:, -1, :]], dim=-1)
        return tokens, self.summary(pooled)


class VariateTokenEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_vars: int,
        per_var_feature_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_ratio: int,
        dropout: float,
        stochastic_depth: float,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_vars = int(n_vars)
        self.per_var_feature_dim = int(per_var_feature_dim)
        self.input_proj = nn.Linear(self.seq_len * self.per_var_feature_dim, d_model)
        self.positional = nn.Parameter(torch.zeros(1, n_vars, d_model))
        self.layers = nn.ModuleList()
        drop_rates = torch.linspace(0.0, float(stochastic_depth), steps=max(int(n_layers), 1)).tolist()
        for idx in range(max(int(n_layers), 1)):
            self.layers.append(
                AttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    drop_path=drop_rates[idx],
                )
            )
        self.summary = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, per_var_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        per_var_features: [B, L, C, F]
        """
        if per_var_features.ndim != 4:
            raise ValueError(f"Expected 4D input [B, L, C, F], got {tuple(per_var_features.shape)}")
        bsz, seq_len, n_vars, feat_dim = per_var_features.shape
        if seq_len != self.seq_len or n_vars != self.n_vars or feat_dim != self.per_var_feature_dim:
            raise ValueError(
                "Unexpected per-variable feature shape. "
                f"Expected [B, {self.seq_len}, {self.n_vars}, {self.per_var_feature_dim}], "
                f"got [B, {seq_len}, {n_vars}, {feat_dim}]"
            )
        tokens = per_var_features.permute(0, 2, 1, 3).reshape(bsz, n_vars, seq_len * feat_dim)
        tokens = self.input_proj(tokens) + self.positional
        for layer in self.layers:
            tokens = layer(tokens)
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)
        return tokens, self.summary(pooled)


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        kernel_size = int(max(3, kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        pad = (self.kernel_size - 1) // 2
        x_t = x.transpose(1, 2)
        front = x_t[:, :, :1].repeat(1, 1, pad)
        end = x_t[:, :, -1:].repeat(1, 1, pad)
        x_t = torch.cat([front, x_t, end], dim=-1)
        out = self.pool(x_t)
        return out.transpose(1, 2)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class ScaleFrequencyEncoder(nn.Module):
    def __init__(
        self,
        n_vars: int,
        seq_len: int,
        d_model: int,
        dropout: float,
        scales: Iterable[int] = (1, 2, 4),
        spectral_topk: int = 8,
    ) -> None:
        super().__init__()
        self.n_vars = int(n_vars)
        self.seq_len = int(seq_len)
        self.scales = tuple(int(max(1, s)) for s in scales)
        self.spectral_topk = int(max(1, spectral_topk))
        kernel_size = max(5, (seq_len // 16) * 2 + 1)
        self.decomp = SeriesDecomposition(kernel_size)
        self.token_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(n_vars * 4 + self.spectral_topk),
                    nn.Linear(n_vars * 4 + self.spectral_topk, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in self.scales
            ]
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.summary = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _pool_scale(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        if scale <= 1:
            return x
        x_t = x.transpose(1, 2)
        pooled = F.avg_pool1d(x_t, kernel_size=scale, stride=scale, ceil_mode=True)
        return pooled.transpose(1, 2)

    def _spectral_summary(self, seasonal: torch.Tensor) -> torch.Tensor:
        # seasonal: [B, L, C]
        spectrum = torch.fft.rfft(seasonal.transpose(1, 2), dim=-1).abs().mean(dim=1)
        topk = min(self.spectral_topk, int(spectrum.shape[-1]))
        values, _ = torch.topk(spectrum, k=topk, dim=-1)
        if topk < self.spectral_topk:
            pad = torch.zeros(
                seasonal.shape[0],
                self.spectral_topk - topk,
                device=seasonal.device,
                dtype=seasonal.dtype,
            )
            values = torch.cat([values, pad], dim=-1)
        return values

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale_tokens = []
        for scale, proj in zip(self.scales, self.token_proj):
            x_scale = self._pool_scale(x, scale)
            seasonal, trend = self.decomp(x_scale)
            summary = torch.cat(
                [
                    trend.mean(dim=1),
                    trend[:, -1, :],
                    seasonal.mean(dim=1),
                    seasonal.std(dim=1, unbiased=False),
                    self._spectral_summary(seasonal),
                ],
                dim=-1,
            )
            scale_tokens.append(proj(summary))
        tokens = torch.stack(scale_tokens, dim=1)
        tokens = self.mix(tokens)
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)
        return tokens, self.summary(pooled)


class TailBoundaryEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        max_steps: int,
        dropout: float,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [
                TemporalMixerBlock(
                    d_model=d_model,
                    ffn_ratio=2,
                    dropout=dropout,
                    drop_path=0.0,
                    kernel_size=5 + 2 * idx,
                )
                for idx in range(max(int(n_layers), 1))
            ]
        )
        self.summary = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        steps = min(int(x.shape[1]), self.max_steps)
        tokens = self.input_proj(x[:, -steps:, :])
        for layer in self.layers:
            tokens = layer(tokens)
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)
        return tokens, self.summary(pooled)


class TopKRouter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_experts: int,
        dropout: float,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(max(1, min(top_k, num_experts)))
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        self.last_logits: torch.Tensor | None = None
        self.last_importance: torch.Tensor | None = None
        self.last_load: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        self.last_logits = logits
        if self.top_k >= self.num_experts:
            weights = torch.softmax(logits, dim=-1)
        else:
            topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)
            sparse_logits = torch.full_like(logits, float("-inf"))
            sparse_logits.scatter_(dim=-1, index=topk_idx, src=topk_vals)
            weights = torch.softmax(sparse_logits, dim=-1)
        with torch.no_grad():
            self.last_importance = weights.mean(dim=0)
            self.last_load = (weights > 0).float().mean(dim=0)
        return weights


class VariableAwareForecastHead(nn.Module):
    def __init__(
        self,
        state_dim: int,
        token_dim: int,
        hidden_dim: int,
        out_len: int,
        n_vars: int,
        rank: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.out_len = int(out_len)
        self.n_vars = int(n_vars)
        self.rank = int(rank)
        self.time_mlp = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_len * rank),
        )
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rank),
        )
        self.bias = nn.Parameter(torch.zeros(out_len, n_vars))

    def forward(self, state: torch.Tensor, var_tokens: torch.Tensor) -> torch.Tensor:
        if var_tokens.shape[1] != self.n_vars:
            var_tokens = F.interpolate(
                var_tokens.transpose(1, 2),
                size=self.n_vars,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        time_basis = self.time_mlp(state).view(state.shape[0], self.out_len, self.rank)
        channel_basis = self.channel_mlp(var_tokens).view(var_tokens.shape[0], self.n_vars, self.rank)
        output = torch.einsum("blr,bcr->blc", time_basis, channel_basis) / math.sqrt(float(self.rank))
        return output + self.bias


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, max(1, int(out_dim))),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
