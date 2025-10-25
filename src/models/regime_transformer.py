"""Transformer model with dual heads for market regime analysis."""

import torch
from torch import nn


class MarketRegimeTransformer(nn.Module):
    """Minimal transformer skeleton with regression and classification heads."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        num_encoder_layers: int,
        num_regimes: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.classification_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_regimes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """Run a forward pass and return both regression and classification outputs."""
        embedded = self.input_projection(x)
        encoded = self.encoder(embedded)
        pooled = encoded.mean(dim=1)
        regression = self.regression_head(pooled).squeeze(-1)
        classification = self.classification_head(pooled)
        return {
            "regression": regression,
            "classification": classification,
        }
