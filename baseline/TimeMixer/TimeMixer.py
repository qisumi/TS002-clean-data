import math

import torch
import torch.nn as nn


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


class DFTSeriesDecomp(nn.Module):
    def __init__(self, top_k: int = 5) -> None:
        super().__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xf = torch.fft.rfft(x, dim=1)
        freq = torch.abs(xf)
        freq[:, 0, :] = 0
        top_k = min(self.top_k, max(freq.shape[1] - 1, 1))
        threshold = torch.topk(freq, top_k, dim=1).values[:, -1:, :]
        xf = torch.where(freq >= threshold, xf, torch.zeros_like(xf))
        season = torch.fft.irfft(xf, n=x.shape[1], dim=1)
        return season, x - season


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False, subtract_last: bool = False, non_norm: bool = False) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if self.non_norm:
            return x
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
        raise ValueError(f"Unsupported Normalize mode: {mode}")


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbeddingWoPos(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x))


class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.down_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** idx),
                        configs.seq_len // (configs.down_sampling_window ** (idx + 1)),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (idx + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (idx + 1)),
                    ),
                )
                for idx in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list: list[torch.Tensor]) -> list[torch.Tensor]:
        out_high = season_list[0]
        out_low = season_list[1]
        outputs = [out_high.permute(0, 2, 1)]
        for idx in range(len(season_list) - 1):
            out_low = out_low + self.down_sampling_layers[idx](out_high)
            out_high = out_low
            if idx + 2 <= len(season_list) - 1:
                out_low = season_list[idx + 2]
            outputs.append(out_high.permute(0, 2, 1))
        return outputs


class MultiScaleTrendMixing(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.up_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (idx + 1)),
                        configs.seq_len // (configs.down_sampling_window ** idx),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** idx),
                        configs.seq_len // (configs.down_sampling_window ** idx),
                    ),
                )
                for idx in reversed(range(configs.down_sampling_layers))
            ]
        )

    def forward(self, trend_list: list[torch.Tensor]) -> list[torch.Tensor]:
        reversed_trends = trend_list[::-1]
        out_low = reversed_trends[0]
        out_high = reversed_trends[1]
        outputs = [out_low.permute(0, 2, 1)]
        for idx in range(len(reversed_trends) - 1):
            out_high = out_high + self.up_sampling_layers[idx](out_low)
            out_low = out_high
            if idx + 2 <= len(reversed_trends) - 1:
                out_high = reversed_trends[idx + 2]
            outputs.append(out_low.permute(0, 2, 1))
        return outputs[::-1]


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.channel_independence = int(configs.channel_independence)
        if configs.decomp_method == "moving_avg":
            self.decomposition = SeriesDecomp(int(configs.moving_avg))
        elif configs.decomp_method == "dft_decomp":
            self.decomposition = DFTSeriesDecomp(int(configs.top_k))
        else:
            raise ValueError(f"Unsupported decomposition method: {configs.decomp_method}")

        if self.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(int(configs.d_model), int(configs.d_ff)),
                nn.GELU(),
                nn.Linear(int(configs.d_ff), int(configs.d_model)),
            )
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)
        self.out_cross_layer = nn.Sequential(
            nn.Linear(int(configs.d_model), int(configs.d_ff)),
            nn.GELU(),
            nn.Linear(int(configs.d_ff), int(configs.d_model)),
        )

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        lengths = [x.shape[1] for x in x_list]
        season_list: list[torch.Tensor] = []
        trend_list: list[torch.Tensor] = []
        for x in x_list:
            season, trend = self.decomposition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)
        outputs: list[torch.Tensor] = []
        for original, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, lengths):
            out = out_season + out_trend
            if self.channel_independence == 1:
                out = original + self.out_cross_layer(out)
            outputs.append(out[:, :length, :])
        return outputs


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.down_sampling_window = int(getattr(configs, "down_sampling_window", 2))
        self.down_sampling_layers = int(getattr(configs, "down_sampling_layers", 0))
        self.down_sampling_method = str(getattr(configs, "down_sampling_method", "avg"))
        self.channel_independence = int(getattr(configs, "channel_independence", 1))
        self.layer = int(getattr(configs, "e_layers", 2))
        self.enc_in = int(getattr(configs, "enc_in", 7))
        self.c_out = int(getattr(configs, "c_out", self.enc_in))
        self.use_norm = int(getattr(configs, "use_norm", 1))

        self.configs = configs
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs) for _ in range(self.layer)])
        if self.channel_independence == 1:
            self.enc_embedding = DataEmbeddingWoPos(c_in=1, d_model=int(configs.d_model), dropout=float(configs.dropout))
        else:
            self.enc_embedding = DataEmbeddingWoPos(
                c_in=int(configs.enc_in),
                d_model=int(configs.d_model),
                dropout=float(configs.dropout),
            )
        self.normalize_layers = nn.ModuleList(
            [
                Normalize(self.enc_in, affine=True, non_norm=(self.use_norm == 0))
                for _ in range(self.down_sampling_layers + 1)
            ]
        )
        self.predict_layers = nn.ModuleList(
            [
                nn.Linear(
                    self.seq_len // (self.down_sampling_window ** idx),
                    self.pred_len,
                )
                for idx in range(self.down_sampling_layers + 1)
            ]
        )
        if self.channel_independence == 1:
            self.projection_layer = nn.Linear(int(configs.d_model), 1, bias=True)
        else:
            self.projection_layer = nn.Linear(int(configs.d_model), self.c_out, bias=True)
            self.out_res_layers = nn.ModuleList(
                [
                    nn.Linear(
                        self.seq_len // (self.down_sampling_window ** idx),
                        self.seq_len // (self.down_sampling_window ** idx),
                    )
                    for idx in range(self.down_sampling_layers + 1)
                ]
            )
            self.regression_layers = nn.ModuleList(
                [
                    nn.Linear(
                        self.seq_len // (self.down_sampling_window ** idx),
                        self.pred_len,
                    )
                    for idx in range(self.down_sampling_layers + 1)
                ]
            )

    def _down_pool(self) -> nn.Module | None:
        if self.down_sampling_method == "avg":
            return nn.AvgPool1d(self.down_sampling_window)
        if self.down_sampling_method == "max":
            return nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        if self.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            return nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.enc_in,
                kernel_size=3,
                padding=padding,
                stride=self.down_sampling_window,
                padding_mode="circular",
                bias=False,
            )
        return None

    def _multi_scale_process_inputs(self, x_enc: torch.Tensor) -> list[torch.Tensor]:
        down_pool = self._down_pool()
        if down_pool is None:
            return [x_enc]
        x_enc_ori = x_enc.permute(0, 2, 1)
        outputs = [x_enc]
        for _ in range(self.down_sampling_layers):
            x_enc_ori = down_pool(x_enc_ori)
            outputs.append(x_enc_ori.permute(0, 2, 1))
        return outputs

    def _pre_enc(self, x_list: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        if self.channel_independence == 1:
            return x_list, None
        out1_list: list[torch.Tensor] = []
        out2_list: list[torch.Tensor] = []
        preprocess = SeriesDecomp(int(getattr(self.configs, "moving_avg", 25)))
        for x in x_list:
            x_1, x_2 = preprocess(x)
            out1_list.append(x_1)
            out2_list.append(x_2)
        return out1_list, out2_list

    def _out_projection(self, dec_out: torch.Tensor, idx: int, out_res: torch.Tensor) -> torch.Tensor:
        dec_out = self.projection_layer(dec_out)
        out_res = self.out_res_layers[idx](out_res.permute(0, 2, 1))
        out_res = self.regression_layers[idx](out_res).permute(0, 2, 1)
        return dec_out + out_res

    def _future_multi_mixing(
        self,
        batch_size: int,
        enc_out_list: list[torch.Tensor],
        x_list: tuple[list[torch.Tensor], list[torch.Tensor] | None],
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        if self.channel_independence == 1:
            series_list = x_list[0]
            for idx, enc_out in enumerate(enc_out_list):
                dec_out = self.predict_layers[idx](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(batch_size, self.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                outputs.append(dec_out)
        else:
            assert x_list[1] is not None
            for idx, (enc_out, out_res) in enumerate(zip(enc_out_list, x_list[1])):
                dec_out = self.predict_layers[idx](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                outputs.append(self._out_projection(dec_out, idx, out_res))
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_multi = self._multi_scale_process_inputs(x)
        x_list: list[torch.Tensor] = []
        for idx, x_item in enumerate(x_multi):
            batch_size, steps, n_vars = x_item.shape
            x_item = self.normalize_layers[idx](x_item, "norm")
            if self.channel_independence == 1:
                x_item = x_item.permute(0, 2, 1).reshape(batch_size * n_vars, steps, 1)
            x_list.append(x_item)

        encoded_inputs = self._pre_enc(x_list)
        enc_out_list: list[torch.Tensor] = []
        for x_item in encoded_inputs[0]:
            enc_out_list.append(self.enc_embedding(x_item))

        for idx in range(self.layer):
            enc_out_list = self.pdm_blocks[idx](enc_out_list)

        dec_out_list = self._future_multi_mixing(batch_size=x.shape[0], enc_out_list=enc_out_list, x_list=encoded_inputs)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        return self.normalize_layers[0](dec_out, "denorm")
