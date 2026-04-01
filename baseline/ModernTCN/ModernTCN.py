import math

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, subtract_last: bool = False) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            dims = tuple(range(1, x.ndim - 1))
            if self.subtract_last:
                self.last = x[:, -1:, :].detach()
            else:
                self.mean = torch.mean(x, dim=dims, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dims, keepdim=True, unbiased=False) + self.eps).detach()
            if self.subtract_last:
                x = x - self.last
            else:
                x = x - self.mean
            x = x / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if mode == "denorm":
            if self.affine:
                x = x - self.affine_bias
                x = x / (self.affine_weight + self.eps * self.eps)
            x = x * self.stdev
            if self.subtract_last:
                return x + self.last
            return x + self.mean
        raise ValueError(f"Unsupported RevIN mode: {mode}")


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class ReparamLargeKernelConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: int | None,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.large = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.large_bn = nn.BatchNorm1d(out_channels)
        if small_kernel is not None:
            self.small = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=small_kernel,
                stride=stride,
                padding=small_kernel // 2,
                groups=groups,
                bias=False,
            )
            self.small_bn = nn.BatchNorm1d(out_channels)
        else:
            self.small = None
            self.small_bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.large_bn(self.large(x))
        if self.small is not None and self.small_bn is not None:
            out = out + self.small_bn(self.small(x))
        return out


class Block(nn.Module):
    def __init__(self, d_model: int, ffn_ratio: int, n_vars: int, large_size: int, small_size: int, dropout: float) -> None:
        super().__init__()
        d_ff = d_model * ffn_ratio
        self.dw = ReparamLargeKernelConv(
            in_channels=n_vars * d_model,
            out_channels=n_vars * d_model,
            kernel_size=large_size,
            stride=1,
            groups=n_vars * d_model,
            small_kernel=small_size,
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.ffn1_pw1 = nn.Conv1d(n_vars * d_model, n_vars * d_ff, kernel_size=1, groups=n_vars)
        self.ffn1_pw2 = nn.Conv1d(n_vars * d_ff, n_vars * d_model, kernel_size=1, groups=n_vars)
        self.ffn2_pw1 = nn.Conv1d(n_vars * d_model, n_vars * d_ff, kernel_size=1, groups=d_model)
        self.ffn2_pw2 = nn.Conv1d(n_vars * d_ff, n_vars * d_model, kernel_size=1, groups=d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        bsz, n_vars, d_model, steps = x.shape
        x = x.reshape(bsz, n_vars * d_model, steps)
        x = self.dw(x)
        x = x.reshape(bsz * n_vars, d_model, steps)
        x = self.norm(x)
        x = x.reshape(bsz, n_vars * d_model, steps)
        x = self.dropout(self.ffn1_pw1(x))
        x = self.activation(x)
        x = self.dropout(self.ffn1_pw2(x))
        x = x.reshape(bsz, n_vars, d_model, steps)

        x = x.permute(0, 2, 1, 3).reshape(bsz, d_model * n_vars, steps)
        x = self.dropout(self.ffn2_pw1(x))
        x = self.activation(x)
        x = self.dropout(self.ffn2_pw2(x))
        x = x.reshape(bsz, d_model, n_vars, steps).permute(0, 2, 1, 3)
        return residual + x


class Stage(nn.Module):
    def __init__(self, num_blocks: int, d_model: int, ffn_ratio: int, n_vars: int, large_size: int, small_size: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    ffn_ratio=ffn_ratio,
                    n_vars=n_vars,
                    large_size=large_size,
                    small_size=small_size,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class FlattenHead(nn.Module):
    def __init__(self, n_vars: int, input_dim: int, target_window: int, head_dropout: float) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(input_dim, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.dropout(self.linear(x))


class ModernTCNCore(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        patch_size: int,
        patch_stride: int,
        downsample_ratio: int,
        num_blocks: list[int],
        large_size: list[int],
        small_size: list[int],
        dims: list[int],
        ffn_ratio: int,
        dropout: float,
        head_dropout: float,
        revin: bool,
        affine: bool,
        subtract_last: bool,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio
        self.num_stage = len(num_blocks)
        self.n_vars = n_vars
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(num_features=n_vars, affine=affine, subtract_last=subtract_last)

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
                nn.BatchNorm1d(dims[0]),
            )
        )
        for idx in range(self.num_stage - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.BatchNorm1d(dims[idx]),
                    nn.Conv1d(dims[idx], dims[idx + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
                )
            )

        self.stages = nn.ModuleList(
            [
                Stage(
                    num_blocks=int(num_blocks[idx]),
                    d_model=int(dims[idx]),
                    ffn_ratio=int(ffn_ratio),
                    n_vars=n_vars,
                    large_size=int(large_size[idx]),
                    small_size=int(small_size[idx]),
                    dropout=dropout,
                )
                for idx in range(self.num_stage)
            ]
        )

        patch_num = max(1, math.ceil(seq_len / patch_stride))
        last_steps = max(1, math.ceil(patch_num / (downsample_ratio ** max(self.num_stage - 1, 0))))
        head_nf = int(dims[self.num_stage - 1]) * last_steps
        self.head = FlattenHead(n_vars=n_vars, input_dim=head_nf, target_window=pred_len, head_dropout=head_dropout)

    def _forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_vars, length = x.shape
        x = x.unsqueeze(-2)
        for idx in range(self.num_stage):
            bsz, n_vars, dim, steps = x.shape
            x = x.reshape(bsz * n_vars, dim, steps)
            if idx == 0 and self.patch_size != self.patch_stride:
                pad_len = self.patch_size - self.patch_stride
                x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad_len)], dim=-1)
            elif idx > 0 and steps % self.downsample_ratio != 0:
                pad_len = self.downsample_ratio - (steps % self.downsample_ratio)
                x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            x = self.downsample_layers[idx](x)
            _, dim, steps = x.shape
            x = x.reshape(bsz, n_vars, dim, steps)
            x = self.stages[idx](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin:
            x = self.revin_layer(x.permute(0, 2, 1), "norm").permute(0, 2, 1)
        x = self._forward_feature(x)
        x = self.head(x)
        if self.revin:
            x = self.revin_layer(x.permute(0, 2, 1), "denorm").permute(0, 2, 1)
        return x


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.decomposition = bool(getattr(configs, "decomposition", False))
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.n_vars = int(configs.enc_in)
        self.kernel_size = int(getattr(configs, "kernel_size", 25))
        self.decomp_module = SeriesDecomp(self.kernel_size)

        kwargs = {
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "n_vars": self.n_vars,
            "patch_size": int(getattr(configs, "patch_size", 8)),
            "patch_stride": int(getattr(configs, "patch_stride", 4)),
            "downsample_ratio": int(getattr(configs, "downsample_ratio", 2)),
            "num_blocks": list(getattr(configs, "num_blocks", [1])),
            "large_size": list(getattr(configs, "large_size", [51])),
            "small_size": list(getattr(configs, "small_size", [5])),
            "dims": list(getattr(configs, "dims", [64])),
            "ffn_ratio": int(getattr(configs, "ffn_ratio", 1)),
            "dropout": float(getattr(configs, "dropout", 0.3)),
            "head_dropout": float(getattr(configs, "head_dropout", 0.0)),
            "revin": bool(getattr(configs, "revin", True)),
            "affine": bool(getattr(configs, "affine", False)),
            "subtract_last": bool(getattr(configs, "subtract_last", False)),
        }

        if self.decomposition:
            self.model_res = ModernTCNCore(**kwargs)
            self.model_trend = ModernTCNCore(**kwargs)
        else:
            self.model = ModernTCNCore(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            return self.model_res(res_init.permute(0, 2, 1)).permute(0, 2, 1) + self.model_trend(
                trend_init.permute(0, 2, 1)
            ).permute(0, 2, 1)
        return self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
