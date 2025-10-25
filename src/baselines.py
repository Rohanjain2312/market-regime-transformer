import torch
import torch.nn as nn

def naive_baseline(y_series):
    # Predict tomorrow = today (return)
    return y_series.shift(1)

def seasonal_naive_baseline(y_series, period=252):
    return y_series.shift(1).rolling(period).mean()

class LSTMBaseline(nn.Module):
    def __init__(self, d_in, hidden=128, layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, num_layers=layers, dropout=dropout if layers>1 else 0, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))

    def forward(self, x):  # [B,T,F]
        o, _ = self.lstm(x)
        h = o[:, -1, :]
        return self.head(h)  # [B,1]
