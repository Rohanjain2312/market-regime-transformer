import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, X, y_reg, y_cls, window=60, horizon=1):
        self.X = X.values.astype("float32")
        self.y_reg = y_reg.values.astype("float32")
        self.y_cls = y_cls.values.astype("int64")
        self.window = window
        self.horizon = horizon
        self.valid_idx = []
        for i in range(len(self.X) - window - horizon + 1):
            # y at i+window+(horizon-1). For horizon=1, next-day.
            if not (any(map(lambda v: v!=v, self.X[i:i+window].ravel()))):
                self.valid_idx.append(i)

    def __len__(self): return len(self.valid_idx)

    def __getitem__(self, idx):
        i = self.valid_idx[idx]
        x = self.X[i:i+self.window]              # [T, F]
        y_r = self.y_reg[i+self.window+self.horizon-1]
        y_c = self.y_cls[i+self.window-1]        # regime at prediction time (current regime)
        x = torch.tensor(x)                      # [T, F]
        # Transformer expects [T, B, D] -> batch_first False
        return x, torch.tensor([y_r], dtype=torch.float32), torch.tensor(y_c, dtype=torch.long)
