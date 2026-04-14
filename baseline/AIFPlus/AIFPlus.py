from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


@dataclass
class AIFPlusConfig:
    seq_len: int
    pred_len: int
    enc_in: int
    horizon_vocab_size: int = 1
    d_model: int = 256
    dropout: float = 0.05
    n_heads: int = 8
    n_patch_layers: int = 2
    n_decoder_layers: int = 2
    ffn_ratio: int = 4
    use_diff_branch: bool = True
    use_channel_context: bool = True
    use_residual_branch: bool = True

    # V8 small-clean controls
    use_linear_head: bool = False
    use_periodic_branch: bool = True
    use_frequency_branch: bool = True
    use_state_channel_mixer: bool = False
    state_channel_heads: int = 4
    deep_residual_max: float = 1.0
    linear_head_use_last_value: bool = True

    patch_len_small: int = 8
    patch_stride_small: int = 4
    patch_len_large: int = 16
    patch_stride_large: int = 8
    patch_jitter: bool = True

    periods: tuple[int, ...] = (24, 48, 96)
    queries_per_period: int = 2
    spectral_topk: int = 8

    residual_hidden: int = 32
    lambda_res_max: float = 0.03

    activation_checkpointing: bool = True
    bc_chunk_size: int = 1024
    eps: float = 1e-5


def _compatible_heads(requested_heads: int, d_model: int) -> int:
    requested_heads = int(max(1, requested_heads))
    d_model = int(max(1, d_model))
    for heads in range(min(requested_heads, d_model), 0, -1):
        if d_model % heads == 0:
            return heads
    return 1


def _apply_with_checkpoint(function: Any, *inputs: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and torch.is_grad_enabled():
        return activation_checkpoint(function, *inputs, use_reentrant=False)
    return function(*inputs)


class RevIN(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.affine = bool(affine)
        self.eps = float(eps)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, 1, self.num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, self.num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        x_hat = (x - mean) / std
        if self.affine:
            x_hat = x_hat * self.weight + self.bias
        return x_hat, (mean, std)

    def denormalize(self, x_hat: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mean, std = stats
        if self.affine:
            x_hat = (x_hat - self.bias) / self.weight.clamp_min(self.eps)
        return x_hat * std + mean


class StochasticDepth(nn.Module):
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(max(0.0, min(1.0, p)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


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


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_ratio: int, dropout: float, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=_compatible_heads(n_heads, d_model),
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model * ffn_ratio, dropout)
        self.drop_path1 = StochasticDepth(drop_path)
        self.drop_path2 = StochasticDepth(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_ratio: int, dropout: float, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=_compatible_heads(n_heads, d_model),
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model * ffn_ratio, dropout)
        self.drop_path1 = StochasticDepth(drop_path)
        self.drop_path2 = StochasticDepth(drop_path)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(self.norm_q(query), self.norm_kv(context), self.norm_kv(context), need_weights=False)
        query = query + self.drop_path1(out)
        query = query + self.drop_path2(self.ffn(self.norm_ffn(query)))
        return query


class ChannelMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChannelContext(nn.Module):
    def __init__(self, seq_len: int, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=int(seq_len),
            num_heads=_compatible_heads(n_heads, seq_len),
            batch_first=True,
            dropout=dropout,
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        x_bc = x_norm.permute(0, 2, 1)
        ctx, _ = self.attn(x_bc, x_bc, x_bc, need_weights=False)
        ctx = self.mix(ctx)
        x_aug = x_bc + ctx
        state = self.state_proj(x_aug.reshape(-1, x_aug.shape[-1]))
        return {"x_aug": x_aug.permute(0, 2, 1), "state": state}


class ChannelStateEncoder(nn.Module):
    """Strictly channel-independent context encoder."""

    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.state_proj = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        x_bc = x_norm.permute(0, 2, 1)
        state = self.state_proj(x_bc.reshape(-1, x_bc.shape[-1]))
        return {"x_aug": x_norm, "state": state}


class PatchBranch(nn.Module):
    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        d_model: int,
        patch_len: int,
        patch_stride: int,
        n_layers: int,
        n_heads: int,
        ffn_ratio: int,
        dropout: float,
        use_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.in_channels = int(in_channels)
        self.d_model = int(d_model)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.use_checkpointing = bool(use_checkpointing)

        self.proj = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.d_model,
            kernel_size=self.patch_len,
            stride=self.patch_stride,
            padding=self.patch_len // 2,
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.seq_len)
            token_len = int(self.proj(dummy).shape[-1])
        self.pos = nn.Parameter(torch.zeros(1, token_len, self.d_model))
        self.type_embed = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    drop_path=0.0,
                )
                for _ in range(max(1, int(n_layers)))
            ]
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _apply_jitter(self, x: torch.Tensor, enabled: bool) -> torch.Tensor:
        if not enabled:
            return x
        max_offset = max(self.patch_stride - 1, 0)
        if max_offset == 0:
            return x
        offset = int(torch.randint(0, max_offset + 1, (1,), device=x.device).item())
        if offset == 0:
            return x
        x = F.pad(x, (offset, 0), mode="replicate")
        return x[..., :-offset]

    def forward(self, x: torch.Tensor, patch_jitter: bool = False) -> dict[str, torch.Tensor]:
        x = self._apply_jitter(x, enabled=patch_jitter and self.training)
        tokens = self.proj(x).transpose(1, 2)
        tokens = tokens + self.pos[:, : tokens.shape[1], :] + self.type_embed
        for layer in self.layers:
            tokens = _apply_with_checkpoint(layer, tokens, enabled=self.use_checkpointing)
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)
        state = self.state_proj(pooled)
        return {"tokens": tokens, "state": state}


class PeriodicTokenEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        periods: Sequence[int],
        queries_per_period: int,
        n_layers: int,
        ffn_ratio: int,
        dropout: float,
        use_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.periods = tuple(int(max(1, p)) for p in periods)
        self.queries_per_period = int(max(1, queries_per_period))
        self.use_checkpointing = bool(use_checkpointing)
        self.query_bank = nn.Parameter(torch.zeros(1, self.queries_per_period, self.d_model))
        self.period_embed = nn.Embedding(max(len(self.periods), 1), self.d_model)
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    drop_path=0.0,
                )
                for _ in range(max(1, int(n_layers)))
            ]
        )
        self.type_embed = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.state_proj = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _build_queries(self, batch_size: int) -> torch.Tensor:
        queries = []
        for idx, _ in enumerate(self.periods):
            base = self.query_bank.expand(batch_size, -1, -1)
            queries.append(base + self.period_embed.weight[idx].view(1, 1, -1))
        if not queries:
            base = self.query_bank.expand(batch_size, -1, -1)
            queries.append(base)
        return torch.cat(queries, dim=1)

    def forward(self, patch_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        queries = self._build_queries(batch_size=patch_tokens.shape[0])
        queries = queries + self.type_embed
        for layer in self.layers:
            queries = _apply_with_checkpoint(layer, queries, patch_tokens, enabled=self.use_checkpointing)
        pooled = torch.cat([queries.mean(dim=1), queries.max(dim=1).values], dim=-1)
        state = self.state_proj(pooled)
        return {"state": state}


class FrequencyTokenEncoder(nn.Module):
    def __init__(self, d_model: int, topk: int, dropout: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.topk = int(max(1, topk))
        self.proj = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.type_embed = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.state_proj = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, series: torch.Tensor) -> dict[str, torch.Tensor]:
        spectrum = torch.fft.rfft(series, dim=-1)
        magnitude = spectrum.abs()
        if magnitude.shape[-1] > 0:
            magnitude[..., 0] = 0.0
        k = min(self.topk, magnitude.shape[-1])
        values, indices = torch.topk(magnitude, k=k, dim=-1)
        denom = max(int(magnitude.shape[-1]) - 1, 1)
        norm_indices = indices.float() / float(denom)
        feats = torch.stack([torch.log1p(values), norm_indices], dim=-1)
        tokens = self.proj(feats) + self.type_embed
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)
        state = self.state_proj(pooled)
        return {"state": state}


class MultiPatchCIEncoder(nn.Module):
    def __init__(self, config: AIFPlusConfig, feature_dim: int) -> None:
        super().__init__()
        self.branch_small = PatchBranch(
            seq_len=config.seq_len,
            in_channels=feature_dim,
            d_model=config.d_model,
            patch_len=config.patch_len_small,
            patch_stride=config.patch_stride_small,
            n_layers=config.n_patch_layers,
            n_heads=config.n_heads,
            ffn_ratio=config.ffn_ratio,
            dropout=config.dropout,
            use_checkpointing=config.activation_checkpointing,
        )
        self.branch_large = PatchBranch(
            seq_len=config.seq_len,
            in_channels=feature_dim,
            d_model=config.d_model,
            patch_len=config.patch_len_large,
            patch_stride=config.patch_stride_large,
            n_layers=config.n_patch_layers,
            n_heads=config.n_heads,
            ffn_ratio=config.ffn_ratio,
            dropout=config.dropout,
            use_checkpointing=config.activation_checkpointing,
        )
        self.task_proj = nn.Linear(config.d_model, 32)
        self.gate = nn.Sequential(
            nn.LayerNorm(config.d_model * 2 + 32),
            nn.Linear(config.d_model * 2 + 32, 64),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor, task_embed: torch.Tensor, patch_jitter: bool = False) -> dict[str, torch.Tensor]:
        out_small = self.branch_small(x, patch_jitter=patch_jitter)
        out_large = self.branch_large(x, patch_jitter=patch_jitter)
        logits = self.gate(torch.cat([out_small["state"], out_large["state"], self.task_proj(task_embed)], dim=-1))
        weights = torch.softmax(logits, dim=-1)
        state = weights[:, :1] * out_small["state"] + weights[:, 1:2] * out_large["state"]
        return {
            "tokens_main": out_small["tokens"],
            "state_small": out_small["state"],
            "state_large": out_large["state"],
            "state": state,
            "branch_weights": weights,
        }


class HorizonQueryDecoder(nn.Module):
    def __init__(
        self,
        pred_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_ratio: int,
        dropout: float,
        use_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)
        self.use_checkpointing = bool(use_checkpointing)
        self.horizon_queries = nn.Parameter(torch.zeros(1, self.pred_len, self.d_model))
        self.seed_proj = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    drop_path=0.0,
                )
                for _ in range(max(1, int(n_layers)))
            ]
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.out = nn.Linear(self.d_model, 1)

    def forward(self, token_bank: torch.Tensor, global_state: torch.Tensor, task_embed: torch.Tensor) -> torch.Tensor:
        seed = self.seed_proj(torch.cat([global_state, task_embed], dim=-1))
        queries = self.horizon_queries + seed.unsqueeze(1)
        for layer in self.layers:
            queries = _apply_with_checkpoint(layer, queries, token_bank, enabled=self.use_checkpointing)
        pred = self.out(self.norm(queries)).squeeze(-1)
        return pred


class DepthwiseTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)
        self.pw = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x)
        x = F.gelu(self.pw(x))
        x = self.norm(x)
        x = self.dropout(x)
        return x + residual


class TinyResidualBranch(nn.Module):
    def __init__(self, config: AIFPlusConfig, feature_dim: int, task_dim: int) -> None:
        super().__init__()
        self.pred_len = int(config.pred_len)
        self.lambda_res_max = float(max(config.lambda_res_max, 0.0))
        self.hidden = int(config.residual_hidden)
        self.input_proj = nn.Conv1d(feature_dim, self.hidden, kernel_size=5, padding=2)
        self.block = DepthwiseTemporalBlock(self.hidden, kernel_size=5, dropout=config.dropout)
        self.state_proj = nn.Sequential(
            nn.LayerNorm(self.hidden * 2),
            nn.Linear(self.hidden * 2, self.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.pred_head = nn.Sequential(
            nn.LayerNorm(self.hidden + task_dim),
            nn.Linear(self.hidden + task_dim, max(self.hidden * 2, 64)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(max(self.hidden * 2, 64), self.pred_len),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.hidden + 2 + task_dim),
            nn.Linear(self.hidden + 2 + task_dim, max(self.hidden * 2, 64)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(max(self.hidden * 2, 64), 1),
        )

    def forward(self, delta_features: torch.Tensor, uncertainty_series: torch.Tensor, task_embed: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.input_proj(delta_features)
        x = self.block(x)
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)
        state = self.state_proj(pooled)
        uncertainty_stats = torch.stack(
            [uncertainty_series.mean(dim=-1), uncertainty_series.amax(dim=-1)],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_head(torch.cat([state, uncertainty_stats, task_embed], dim=-1))) * self.lambda_res_max
        pred = self.pred_head(torch.cat([state, task_embed], dim=-1))
        return {"pred": pred, "gate": gate, "state": state}


class SharedNLinearHead(nn.Module):
    """Channel-independent direct head, strong on small clean datasets."""

    def __init__(self, seq_len: int, pred_len: int, use_last_value: bool = True) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.use_last_value = bool(use_last_value)
        self.linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, series_bc: torch.Tensor) -> torch.Tensor:
        if self.use_last_value:
            anchor = series_bc[:, -1:].detach()
            centered = series_bc - anchor
            pred = self.linear(centered) + anchor
            return pred
        return self.linear(series_bc)


class TinyStateChannelMixer(nn.Module):
    """Lightweight channel interaction over per-channel states, not full sequences."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=_compatible_heads(n_heads, d_model),
            batch_first=True,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model * 2, dropout)
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, max(d_model // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_model // 2, 32), 1),
        )

    def forward(self, state_bc: torch.Tensor, batch_size: int, n_vars: int) -> torch.Tensor:
        states = state_bc.view(batch_size, n_vars, -1)
        attn_in = self.norm1(states)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        mixed = states + attn_out
        mixed = mixed + self.ffn(self.norm2(mixed))
        alpha = torch.sigmoid(self.gate(torch.cat([states, mixed], dim=-1)))
        out = states + alpha * (mixed - states)
        return out.reshape(batch_size * n_vars, -1)


class DeepResidualGate(nn.Module):
    def __init__(self, d_model: int, task_dim: int, dropout: float, max_scale: float) -> None:
        super().__init__()
        self.max_scale = float(max(max_scale, 0.0))
        hidden = max(d_model // 2, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model + task_dim),
            nn.Linear(d_model + task_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Start with the deep branch as a small correction on top of the linear head.
        final_linear = self.net[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.constant_(final_linear.bias, -2.0)

    def forward(self, deep_state: torch.Tensor, task_embed: torch.Tensor) -> torch.Tensor:
        if self.max_scale <= 0.0:
            return deep_state.new_zeros((deep_state.shape[0], 1))
        gate = torch.sigmoid(self.net(torch.cat([deep_state, task_embed], dim=-1)))
        return gate * self.max_scale


class AIFPlusLoss(nn.Module):
    def __init__(self, mae_weight: float = 0.7, mse_weight: float = 0.3) -> None:
        super().__init__()
        self.mae_weight = float(mae_weight)
        self.mse_weight = float(mse_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        mae = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        loss = self.mae_weight * mae + self.mse_weight * mse
        return {"loss": loss, "mae": mae, "mse": mse}


class AIFPlus(nn.Module):
    """
    AIF-Plus-V8
    - keeps the clean-first V7 training/eval philosophy
    - adds an optional NLinear-style direct head for small clean datasets
    - deep token model becomes a residual corrector on top of the direct head
    - optional lightweight channel-state mixer (useful for tiny multivariate datasets such as exchange_rate)
    - periodic/frequency branches can be disabled dataset-wise to reduce over-parameterization
    """

    def __init__(self, config: AIFPlusConfig) -> None:
        super().__init__()
        self.config = config
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.n_vars = int(config.enc_in)
        self.use_diff_branch = bool(config.use_diff_branch)
        self.use_channel_context = bool(config.use_channel_context)
        self.use_residual_branch = bool(config.use_residual_branch)
        self.use_linear_head = bool(config.use_linear_head)
        self.use_periodic_branch = bool(config.use_periodic_branch)
        self.use_frequency_branch = bool(config.use_frequency_branch)
        self.use_state_channel_mixer = bool(config.use_state_channel_mixer)
        self.bc_chunk_size = max(int(config.bc_chunk_size), 0)

        self.revin = RevIN(num_features=self.n_vars, affine=True, eps=config.eps)
        self.task_embed = nn.Embedding(max(int(config.horizon_vocab_size), 1), int(config.d_model))
        self.feature_dim = 2 if self.use_diff_branch else 1

        if self.use_channel_context:
            self.channel_context = ChannelContext(
                seq_len=config.seq_len,
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout,
            )
        else:
            self.channel_context = ChannelStateEncoder(
                seq_len=config.seq_len,
                d_model=config.d_model,
                dropout=config.dropout,
            )

        self.encoder = MultiPatchCIEncoder(config=config, feature_dim=self.feature_dim)
        self.periodic = (
            PeriodicTokenEncoder(
                d_model=config.d_model,
                n_heads=config.n_heads,
                periods=config.periods,
                queries_per_period=config.queries_per_period,
                n_layers=1,
                ffn_ratio=config.ffn_ratio,
                dropout=config.dropout,
                use_checkpointing=config.activation_checkpointing,
            )
            if self.use_periodic_branch
            else None
        )
        self.frequency = (
            FrequencyTokenEncoder(
                d_model=config.d_model,
                topk=config.spectral_topk,
                dropout=config.dropout,
            )
            if self.use_frequency_branch
            else None
        )
        self.state_channel_mixer = (
            TinyStateChannelMixer(
                d_model=config.d_model,
                n_heads=config.state_channel_heads,
                dropout=config.dropout,
            )
            if self.use_state_channel_mixer
            else None
        )

        summary_tokens = 2 + int(self.use_periodic_branch) + int(self.use_frequency_branch)
        self.summary_type_embed = nn.Parameter(torch.zeros(1, summary_tokens, int(config.d_model)))
        global_state_inputs = 3 + int(self.use_periodic_branch) + int(self.use_frequency_branch)
        self.global_state = ChannelMLP(
            in_dim=config.d_model * global_state_inputs,
            out_dim=config.d_model,
            dropout=config.dropout,
        )
        self.decoder = HorizonQueryDecoder(
            pred_len=config.pred_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_decoder_layers,
            ffn_ratio=config.ffn_ratio,
            dropout=config.dropout,
            use_checkpointing=config.activation_checkpointing,
        )
        self.linear_head = (
            SharedNLinearHead(
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                use_last_value=config.linear_head_use_last_value,
            )
            if self.use_linear_head
            else None
        )
        self.deep_gate = (
            DeepResidualGate(
                d_model=config.d_model,
                task_dim=config.d_model,
                dropout=config.dropout,
                max_scale=config.deep_residual_max,
            )
            if self.use_linear_head
            else None
        )
        self.residual = (
            TinyResidualBranch(
                config=config,
                feature_dim=self.feature_dim,
                task_dim=config.d_model,
            )
            if self.use_residual_branch
            else None
        )

    @staticmethod
    def _first_difference(x: torch.Tensor) -> torch.Tensor:
        diff = x[:, 1:, :] - x[:, :-1, :]
        pad = torch.zeros_like(x[:, :1, :])
        return torch.cat([pad, diff], dim=1)

    def _build_clean_features(self, x_norm: torch.Tensor) -> torch.Tensor:
        features = [x_norm.permute(0, 2, 1).unsqueeze(2)]
        if self.use_diff_branch:
            diff = self._first_difference(x_norm).permute(0, 2, 1).unsqueeze(2)
            features.append(diff)
        feat = torch.cat(features, dim=2)
        return feat.reshape(x_norm.shape[0] * x_norm.shape[2], feat.shape[2], x_norm.shape[1])

    def _build_delta_features(self, delta_norm: torch.Tensor) -> torch.Tensor:
        features = [delta_norm.permute(0, 2, 1).unsqueeze(2)]
        if self.use_diff_branch:
            diff = self._first_difference(delta_norm).permute(0, 2, 1).unsqueeze(2)
            features.append(diff)
        feat = torch.cat(features, dim=2)
        return feat.reshape(delta_norm.shape[0] * delta_norm.shape[2], feat.shape[2], delta_norm.shape[1])

    def _expand_task_embed(self, horizon_id: torch.Tensor) -> torch.Tensor:
        if horizon_id.ndim == 0:
            horizon_id = horizon_id.view(1)
        task = self.task_embed(horizon_id)
        task = task.unsqueeze(1).expand(-1, self.n_vars, -1)
        return task.reshape(task.shape[0] * task.shape[1], task.shape[2])

    def _build_summary_tokens(
        self,
        enc_state: torch.Tensor,
        channel_state: torch.Tensor,
        period_state: torch.Tensor | None,
        freq_state: torch.Tensor | None,
    ) -> torch.Tensor:
        states = [enc_state, channel_state]
        if period_state is not None:
            states.append(period_state)
        if freq_state is not None:
            states.append(freq_state)
        summary = torch.stack(states, dim=1)
        return summary + self.summary_type_embed[:, : summary.shape[1], :]

    def _run_clean_trunk(
        self,
        clean_features: torch.Tensor,
        masked_series_bc: torch.Tensor,
        task_embed: torch.Tensor,
        channel_state_bc: torch.Tensor,
        batch_size: int,
        return_aux: bool,
    ) -> dict[str, torch.Tensor]:
        total = int(clean_features.shape[0])
        chunk_size = total if self.bc_chunk_size <= 0 else min(self.bc_chunk_size, total)
        if self.use_state_channel_mixer:
            chunk_size = total

        pred_deep_chunks: list[torch.Tensor] = []
        global_state_chunks: list[torch.Tensor] = []
        branch_weight_chunks: list[torch.Tensor] | None = [] if return_aux else None

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            enc = self.encoder(
                clean_features[start:end],
                task_embed=task_embed[start:end],
                patch_jitter=bool(self.config.patch_jitter),
            )
            enc_state = enc["state"]
            if self.state_channel_mixer is not None:
                local_count = end - start
                local_batch = max(local_count // self.n_vars, 1)
                enc_state = self.state_channel_mixer(enc_state, batch_size=local_batch, n_vars=self.n_vars)

            period_state = self.periodic(enc["tokens_main"])["state"] if self.periodic is not None else None
            freq_state = self.frequency(masked_series_bc[start:end])["state"] if self.frequency is not None else None
            summary_tokens = self._build_summary_tokens(
                enc_state=enc_state,
                channel_state=channel_state_bc[start:end],
                period_state=period_state,
                freq_state=freq_state,
            )
            token_bank = torch.cat([enc["tokens_main"], summary_tokens], dim=1)
            global_inputs = [enc_state, channel_state_bc[start:end], task_embed[start:end]]
            if period_state is not None:
                global_inputs.append(period_state)
            if freq_state is not None:
                global_inputs.append(freq_state)
            global_state = self.global_state(torch.cat(global_inputs, dim=-1))
            pred_deep = self.decoder(token_bank=token_bank, global_state=global_state, task_embed=task_embed[start:end])
            pred_deep_chunks.append(pred_deep)
            global_state_chunks.append(global_state)
            if branch_weight_chunks is not None:
                branch_weight_chunks.append(enc["branch_weights"])

        outputs = {
            "pred_deep": torch.cat(pred_deep_chunks, dim=0),
            "global_state": torch.cat(global_state_chunks, dim=0),
        }
        if branch_weight_chunks is not None:
            outputs["branch_weights"] = torch.cat(branch_weight_chunks, dim=0)
        return outputs

    def forward(
        self,
        x_raw: torch.Tensor,
        x_masked: torch.Tensor,
        uncertainty: torch.Tensor,
        metadata_num: Optional[torch.Tensor] = None,
        dataset_id: Optional[torch.Tensor] = None,
        support_id: Optional[torch.Tensor] = None,
        horizon_id: Optional[torch.Tensor] = None,
        return_aux: bool = True,
    ) -> dict[str, torch.Tensor]:
        del metadata_num, dataset_id, support_id

        if horizon_id is None:
            horizon_id = torch.zeros(x_raw.shape[0], dtype=torch.long, device=x_raw.device)
        if horizon_id.ndim == 0:
            horizon_id = horizon_id.view(1).expand(x_raw.shape[0])

        batch_size = int(x_raw.shape[0])
        masked_norm, masked_stats = self.revin.normalize(x_masked)
        channel_context = self.channel_context(masked_norm)
        masked_aug = channel_context["x_aug"]
        channel_state_bc = channel_context["state"]

        clean_features = self._build_clean_features(masked_aug)
        task_embed = self._expand_task_embed(horizon_id)
        masked_series_bc = masked_aug.permute(0, 2, 1).reshape(batch_size * self.n_vars, self.seq_len)
        masked_norm_bc = masked_norm.permute(0, 2, 1).reshape(batch_size * self.n_vars, self.seq_len)

        clean_out = self._run_clean_trunk(
            clean_features=clean_features,
            masked_series_bc=masked_series_bc,
            task_embed=task_embed,
            channel_state_bc=channel_state_bc,
            batch_size=batch_size,
            return_aux=return_aux,
        )
        pred_deep_norm_bc = clean_out["pred_deep"]
        if self.linear_head is not None and self.deep_gate is not None:
            pred_linear_norm_bc = self.linear_head(masked_norm_bc)
            deep_gate = self.deep_gate(clean_out["global_state"], task_embed)
            pred_clean_norm_bc = pred_linear_norm_bc + deep_gate * pred_deep_norm_bc
        else:
            pred_linear_norm_bc = pred_deep_norm_bc.new_zeros(pred_deep_norm_bc.shape)
            deep_gate = pred_deep_norm_bc.new_ones((pred_deep_norm_bc.shape[0], 1))
            pred_clean_norm_bc = pred_deep_norm_bc

        mean_masked, std_masked = masked_stats
        if self.use_residual_branch and self.residual is not None:
            delta_norm = (x_raw - x_masked) / std_masked.clamp_min(self.config.eps)
            delta_features = self._build_delta_features(delta_norm)
            uncertainty_bc = uncertainty.permute(0, 2, 1).reshape(batch_size * self.n_vars, self.seq_len)
            residual_out = self.residual(delta_features=delta_features, uncertainty_series=uncertainty_bc, task_embed=task_embed)
            residual_gate = residual_out["gate"]
            residual_term_bc = residual_gate * residual_out["pred"]
            pred_norm_bc = pred_clean_norm_bc + residual_term_bc
        else:
            residual_gate = pred_clean_norm_bc.new_zeros((pred_clean_norm_bc.shape[0], 1))
            residual_term_bc = pred_clean_norm_bc.new_zeros(pred_clean_norm_bc.shape)
            pred_norm_bc = pred_clean_norm_bc

        pred_norm = pred_norm_bc.view(batch_size, self.n_vars, self.pred_len).transpose(1, 2)
        pred = self.revin.denormalize(pred_norm, masked_stats)
        if not return_aux:
            return {"pred": pred}

        pred_clean_norm = pred_clean_norm_bc.view(batch_size, self.n_vars, self.pred_len).transpose(1, 2)
        pred_deep_norm = pred_deep_norm_bc.view(batch_size, self.n_vars, self.pred_len).transpose(1, 2)
        pred_linear_norm = pred_linear_norm_bc.view(batch_size, self.n_vars, self.pred_len).transpose(1, 2)
        residual_norm = residual_term_bc.view(batch_size, self.n_vars, self.pred_len).transpose(1, 2)

        pred_clean = self.revin.denormalize(pred_clean_norm, masked_stats)
        pred_deep = pred_deep_norm * std_masked + mean_masked
        pred_linear = pred_linear_norm * std_masked + mean_masked
        residual = residual_norm * std_masked

        return {
            "pred": pred,
            "pred_clean": pred_clean,
            "pred_deep": pred_deep,
            "pred_linear": pred_linear,
            "pred_residual": residual,
            "lambda_res": residual_gate.view(batch_size, self.n_vars),
            "lambda_deep": deep_gate.view(batch_size, self.n_vars),
            "branch_weights": clean_out.get("branch_weights", pred_clean_norm_bc.new_zeros((batch_size * self.n_vars, 2))).view(batch_size, self.n_vars, 2),
            "task_embed": task_embed.view(batch_size, self.n_vars, -1),
        }

    @torch.no_grad()
    def predict(
        self,
        x_raw: torch.Tensor,
        x_masked: torch.Tensor,
        uncertainty: torch.Tensor,
        metadata_num: Optional[torch.Tensor] = None,
        dataset_id: Optional[torch.Tensor] = None,
        support_id: Optional[torch.Tensor] = None,
        horizon_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(
            x_raw=x_raw,
            x_masked=x_masked,
            uncertainty=uncertainty,
            metadata_num=metadata_num,
            dataset_id=dataset_id,
            support_id=support_id,
            horizon_id=horizon_id,
            return_aux=False,
        )["pred"]
