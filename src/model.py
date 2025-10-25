import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [T, D]

    def forward(self, x):  # x: [T, B, D]
        T = x.size(0)
        return x + self.pe[:T].unsqueeze(1)

class RegimeSwitchingTransformer(nn.Module):
    def __init__(self, d_in, d_model=256, nhead=4, num_layers=3, d_ff=512, dropout=0.1, num_classes=3):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.pe = PositionalEncoding(d_model)
        self.reg_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model//2, num_classes))
        self.regr_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model//2, 1))

    def forward(self, src):  # src: [B, T, F] -> convert to [T, B, F]
        src = src.transpose(0, 1)  # [T, B, F]
        h = self.input_proj(src)   # [T, B, D]
        h = self.pe(h)
        h = self.encoder(h)        # [T, B, D]
        h_last = h[-1]             # [B, D]
        y_reg = self.regr_head(h_last)   # [B, 1]
        y_cls = self.reg_head(h_last)    # [B, C]
        return y_reg, y_cls
