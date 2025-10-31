import torch
import torch.nn as nn

class RegimeTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()

        # project features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # simple learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        bsz, seq_len, _ = x.size()

        x = self.input_proj(x)  # (B, T, d_model)
        x = x + self.pos_embedding[:, :seq_len, :]

        h = self.encoder(x)  # (B, T, d_model)

        # mean pool
        h_pool = h.mean(dim=1)  # (B, d_model)

        logits = self.cls_head(h_pool)
        return logits
