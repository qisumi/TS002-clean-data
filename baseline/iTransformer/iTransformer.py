import torch
import torch.nn as nn
import torch.nn.functional as F


class DataEmbeddingInverted(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x.permute(0, 2, 1)))


class FullAttention(nn.Module):
    def __init__(self, attention_dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        scale = queries.shape[-1] ** -0.5
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,bshd->blhd", attn, values)


class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.inner_attention = FullAttention(attention_dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        queries = self.query_projection(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = self.key_projection(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        values = self.value_projection(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        out = self.inner_attention(queries, keys, values).reshape(bsz, seq_len, -1)
        return self.out_projection(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.attention = AttentionLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(x))
        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, e_layers: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.use_norm = bool(getattr(configs, "use_norm", True))
        d_model = int(getattr(configs, "d_model", 512))
        d_ff = int(getattr(configs, "d_ff", 2048))
        n_heads = int(getattr(configs, "n_heads", 8))
        e_layers = int(getattr(configs, "e_layers", 2))
        dropout = float(getattr(configs, "dropout", 0.1))
        activation = str(getattr(configs, "activation", "gelu"))

        self.enc_embedding = DataEmbeddingInverted(seq_len=self.seq_len, d_model=d_model, dropout=dropout)
        self.encoder = Encoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            dropout=dropout,
            activation=activation,
        )
        self.projector = nn.Linear(d_model, self.pred_len, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        else:
            means = None
            stdev = None

        enc_out = self.enc_embedding(x)
        enc_out = self.encoder(enc_out)
        dec_out = self.projector(enc_out).permute(0, 2, 1)

        if self.use_norm and means is not None and stdev is not None:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        return dec_out[:, -self.pred_len :, :]
