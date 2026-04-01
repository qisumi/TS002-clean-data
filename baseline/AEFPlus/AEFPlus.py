from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


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
    def __init__(self, d_model: int, n_heads: int, ffn_ratio: int, dropout: float, drop_path: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_conv = nn.LayerNorm(d_model)
        self.temporal_conv = MultiKernelTemporalConv(d_model=d_model, dropout=dropout, kernels=(3, 5, 9))
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


class BoundaryEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, max_boundary_steps: int, dropout: float) -> None:
        super().__init__()
        self.max_boundary_steps = int(max_boundary_steps)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.summary = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        steps = min(int(x.shape[1]), self.max_boundary_steps)
        boundary = self.input_proj(x[:, -steps:, :])
        boundary_t = boundary.transpose(1, 2)
        mixed = 0.5 * (self.conv3(boundary_t) + self.conv5(boundary_t))
        tokens = self.norm(F.gelu(mixed).transpose(1, 2))
        tokens = self.dropout(tokens)
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)
        return tokens, self.summary(pooled)


class MetadataTokenEncoder(nn.Module):
    def __init__(
        self,
        metadata_num_dim: int,
        metadata_dim: int,
        artifact_vocab_size: int,
        phase_vocab_size: int,
        severity_vocab_size: int,
        nvar_vocab_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.numeric_mlp = nn.Sequential(
            nn.LayerNorm(metadata_num_dim),
            nn.Linear(metadata_num_dim, metadata_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(metadata_dim * 2, metadata_dim),
        )
        self.artifact_embed = nn.Embedding(max(artifact_vocab_size, 1), metadata_dim)
        self.phase_embed = nn.Embedding(max(phase_vocab_size, 1), metadata_dim)
        self.severity_embed = nn.Embedding(max(severity_vocab_size, 1), metadata_dim)
        self.nvar_embed = nn.Embedding(max(nvar_vocab_size, 1), metadata_dim)
        self.positional = nn.Parameter(torch.zeros(1, 5, metadata_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=metadata_dim,
            nhead=4,
            dim_feedforward=metadata_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.summary = nn.Sequential(
            nn.LayerNorm(metadata_dim * 2),
            nn.Linear(metadata_dim * 2, metadata_dim),
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack(
            [
                self.numeric_mlp(metadata_num),
                self.artifact_embed(artifact_id),
                self.phase_embed(phase_id),
                self.severity_embed(severity_bin_id),
                self.nvar_embed(nvar_bin_id),
            ],
            dim=1,
        )
        tokens = self.transformer(tokens + self.positional)
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)
        return tokens, self.summary(pooled)


class CrossGatedFusion(nn.Module):
    def __init__(self, d_model: int, metadata_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_proj = nn.Linear(metadata_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate_proj = nn.Sequential(
            nn.Linear(metadata_dim, d_model),
            nn.Sigmoid(),
        )
        self.state_norm = nn.LayerNorm(d_model)
        self.token_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        global_tokens: torch.Tensor,
        global_state: torch.Tensor,
        boundary_tokens: torch.Tensor,
        meta_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.query_proj(meta_state).unsqueeze(1)
        exploit_token, _ = self.cross_attn(query, boundary_tokens, boundary_tokens, need_weights=False)
        exploit_token = exploit_token.squeeze(1)
        gate = self.gate_proj(meta_state)
        fused_state = self.state_norm(global_state + gate * exploit_token)
        fused_tokens = global_tokens + self.token_proj(exploit_token).unsqueeze(1)
        return fused_tokens, fused_state, exploit_token


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


class SparseExpert(nn.Module):
    def __init__(self, mode: str, d_model: int, hidden_dim: int, pred_len: int, n_vars: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.mode = str(mode)
        if self.mode == "zero_flat":
            self.boundary_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
            summary_dim = d_model * 2
        elif self.mode == "transition":
            self.boundary_conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=4, dilation=2)
            summary_dim = d_model * 4
        elif self.mode == "repeat":
            self.boundary_conv = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
            summary_dim = d_model * 2
        elif self.mode == "generic":
            self.boundary_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
            summary_dim = d_model * 2
        else:
            raise ValueError(f"Unsupported expert mode: {mode}")
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = FactorizedForecastHead(
            in_dim=d_model * 2,
            hidden_dim=hidden_dim,
            out_len=pred_len,
            n_vars=n_vars,
            rank=rank,
            dropout=dropout,
        )

    def _summarize(self, global_tokens: torch.Tensor, boundary_tokens: torch.Tensor) -> torch.Tensor:
        boundary_t = self.boundary_conv(boundary_tokens.transpose(1, 2)).transpose(1, 2)
        if self.mode == "zero_flat":
            return torch.cat([boundary_t.mean(dim=1), boundary_t.min(dim=1).values], dim=-1)
        if self.mode == "transition":
            return boundary_t[:, -4:, :].reshape(boundary_t.shape[0], -1)
        if self.mode == "repeat":
            return torch.cat([boundary_t.mean(dim=1), boundary_t.max(dim=1).values], dim=-1)
        return torch.cat([global_tokens.mean(dim=1), global_tokens[:, -1, :]], dim=-1)

    def forward(self, global_tokens: torch.Tensor, boundary_tokens: torch.Tensor, fused_state: torch.Tensor) -> torch.Tensor:
        summary = self.summary_proj(self._summarize(global_tokens, boundary_tokens))
        return self.head(torch.cat([fused_state, summary], dim=-1))


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
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


class AEFPlus(nn.Module):
    def __init__(self, config: AEFPlusConfig) -> None:
        super().__init__()
        self.config = config
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.n_vars = int(config.enc_in)
        self.revin = RevIN(num_features=self.n_vars, affine=True)

        self.shared_trunk = SharedTrunk(
            input_dim=self.n_vars,
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
        self.boundary_encoder = BoundaryEncoder(
            input_dim=self.n_vars,
            d_model=int(config.d_model),
            max_boundary_steps=min(int(config.max_boundary_steps), max(4, self.seq_len // 4)),
            dropout=float(config.dropout),
        )
        self.meta_encoder = MetadataTokenEncoder(
            metadata_num_dim=int(config.metadata_num_dim),
            metadata_dim=int(config.metadata_dim),
            artifact_vocab_size=int(config.artifact_vocab_size),
            phase_vocab_size=int(config.phase_vocab_size),
            severity_vocab_size=int(config.severity_vocab_size),
            nvar_vocab_size=int(config.nvar_vocab_size),
            dropout=float(config.dropout),
        )
        self.fusion = CrossGatedFusion(
            d_model=int(config.d_model),
            metadata_dim=int(config.metadata_dim),
            n_heads=int(config.n_heads),
            dropout=float(config.dropout),
        )
        self.horizon_embed = nn.Embedding(max(int(config.horizon_vocab_size), 1), 16)
        self.router_phase_embed = nn.Embedding(max(int(config.phase_vocab_size), 1), 16)
        self.router = nn.Sequential(
            nn.LayerNorm(int(config.metadata_dim) + 16 + 16),
            nn.Linear(int(config.metadata_dim) + 32, int(config.expert_hidden)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.expert_hidden), int(config.num_experts)),
        )
        expert_modes = ["zero_flat", "transition", "repeat", "generic"][: int(config.num_experts)]
        self.experts = nn.ModuleList(
            [
                SparseExpert(
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
        aux_dim = int(config.d_model) + int(config.metadata_dim)
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
        global_tokens, global_state = self.shared_trunk(x_norm)
        boundary_tokens, boundary_state = self.boundary_encoder(x_norm)
        _, meta_state = self.meta_encoder(
            metadata_num=metadata_num,
            artifact_id=artifact_id,
            phase_id=phase_id,
            severity_bin_id=severity_bin_id,
            nvar_bin_id=nvar_bin_id,
        )
        fused_tokens, fused_state, exploit_state = self.fusion(
            global_tokens=global_tokens,
            global_state=global_state + boundary_state,
            boundary_tokens=boundary_tokens,
            meta_state=meta_state,
        )
        router_input = torch.cat(
            [
                meta_state,
                self.horizon_embed(horizon_id),
                self.router_phase_embed(phase_id),
            ],
            dim=-1,
        )
        router_weights = torch.softmax(self.router(router_input), dim=-1)
        expert_preds = torch.stack(
            [expert(fused_tokens, boundary_tokens, fused_state) for expert in self.experts],
            dim=1,
        )
        pred_norm = torch.sum(router_weights.unsqueeze(-1).unsqueeze(-1) * expert_preds, dim=1)
        aux_state = torch.cat([fused_state + exploit_state, meta_state], dim=-1)
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
