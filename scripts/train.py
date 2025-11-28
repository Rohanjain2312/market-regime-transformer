import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from fredapi import Fred
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from hmmlearn.hmm import GaussianHMM

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')

@dataclass
class Config:
    # Data
    tickers: List[str]
    target_ticker: str
    start_date: str
    end_date: str

    # Features
    lookback: int
    horizon: int

    # Model
    d_model: int
    n_heads: int
    num_layers: int
    dropout: float
    num_regimes: int

    # Training
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    patience: int
    lambda_reg: float
    lambda_cls: float

def download_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    print(f"Downloading data for {tickers} from {start} to {end}...")
    data = yf.download(tickers, start=start, end=end)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data

# Feature Engineering Functions
def log_return(series: pd.Series) -> pd.Series:
    return np.log(series).diff()

def rolling_vol(series: pd.Series, window: int = 21) -> pd.Series:
    return series.rolling(window=window, min_periods=window // 2).std()

def rolling_mean(series: pd.Series, window: int = 21) -> pd.Series:
    return series.rolling(window=window, min_periods=window // 2).mean()

def momentum(series: pd.Series, lookback: int = 21) -> pd.Series:
    return series / series.shift(lookback) - 1.0

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    gain = up.ewm(alpha=1 / period, adjust=False).mean()
    loss = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def build_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Process each ticker
    for col in df.columns:
        px = df[col]
        ret = log_return(px)

        # Basic features for all
        out[f'{col}_ret'] = ret
        out[f'{col}_vol_21'] = rolling_vol(ret, 21)
        out[f'{col}_mom_21'] = momentum(px, 21)
        out[f'{col}_mom_63'] = momentum(px, 63)

        # Extra features for target
        if col == target_col:
            out[f'{col}_rsi'] = rsi(px)
            out[f'{col}_sma_50'] = rolling_mean(px, 50)
            out[f'{col}_sma_200'] = rolling_mean(px, 200)

    return out.dropna()

def label_regimes_hmm(df: pd.DataFrame, target_col: str, n_components: int = 3) -> pd.Series:
    """Label regimes using Hidden Markov Model on returns and volatility."""
    ret_col = f'{target_col}_ret'
    vol_col = f'{target_col}_vol_21'
    
    # Prepare data for HMM
    X = df[[ret_col, vol_col]].values
    
    # Fit HMM
    print("Fitting HMM...")
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X)
    
    # Predict states
    states = model.predict(X)
    
    # Map states to Bear (0), Neutral (1), Bull (2) based on mean returns
    state_means = []
    for i in range(n_components):
        mean_ret = X[states == i, 0].mean()
        state_means.append((i, mean_ret))
    
    # Sort by return: lowest return = Bear, highest = Bull
    state_means.sort(key=lambda x: x[1])
    
    mapping = {}
    mapping[state_means[0][0]] = 0  # Bear
    mapping[state_means[1][0]] = 1  # Neutral
    mapping[state_means[2][0]] = 2  # Bull
    
    mapped_states = np.array([mapping[s] for s in states])
    
    return pd.Series(mapped_states, index=df.index)

class MarketDataset(Dataset):
    def __init__(self, X, y_reg, y_cls, lookback):
        self.X = torch.FloatTensor(X)
        self.y_reg = torch.FloatTensor(y_reg)
        self.y_cls = torch.LongTensor(y_cls)
        self.lookback = lookback

    def __len__(self):
        return len(self.X) - self.lookback

    def __getitem__(self, idx):
        return (
            self.X[idx:idx+self.lookback],      # Input sequence
            self.y_reg[idx+self.lookback],      # Next day return
            self.y_cls[idx+self.lookback]       # Next day regime
        )

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :]

class MarketRegimeTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, num_regimes, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Heads
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_regimes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Global average pooling
        x_pool = x.mean(dim=1)

        return {
            'reg': self.reg_head(x_pool).squeeze(-1),
            'cls': self.cls_head(x_pool)
        }

def train_model(model, train_loader, val_loader, config, class_weights=None, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    criterion_reg = nn.MSELoss()
    if class_weights is not None:
        criterion_cls = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion_cls = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Training on {device}...")
    model.to(device)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for X, y_reg, y_cls in train_loader:
            X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
            
            optimizer.zero_grad()
            out = model(X)

            loss_reg = criterion_reg(out['reg'], y_reg)
            loss_cls = criterion_cls(out['cls'], y_cls)

            loss = config.lambda_reg * loss_reg + config.lambda_cls * loss_cls
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y_reg, y_cls in val_loader:
                X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
                
                out = model(X)
                loss_reg = criterion_reg(out['reg'], y_reg)
                loss_cls = criterion_cls(out['cls'], y_cls)
                val_loss += (config.lambda_reg * loss_reg + config.lambda_cls * loss_cls).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model and Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping triggered")
                break

    return history

def main():
    parser = argparse.ArgumentParser(description='Market Regime Transformer Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    config = Config(
        tickers=['^GSPC', 'SPY', '^IXIC', '^VIX'],
        target_ticker='^GSPC',
        start_date='2000-01-01',
        end_date='2024-12-31',
        lookback=60,
        horizon=1,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.2,
        num_regimes=3,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-5,
        patience=15,
        lambda_reg=1.0,
        lambda_cls=2.0
    )

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    raw_data = download_data(config.tickers, config.start_date, config.end_date)

    # 2. Features
    print("Building features...")
    features = build_features(raw_data, config.target_ticker)

    # 3. Labels (HMM)
    print("Generating labels with HMM...")
    regimes = label_regimes_hmm(features, config.target_ticker)

    # 4. Prepare Dataset
    data_df = features.join(regimes.rename('regime'))
    data_df = data_df.dropna()

    # Split
    train_size = int(len(data_df) * 0.7)
    val_size = int(len(data_df) * 0.15)
    test_size = len(data_df) - train_size - val_size

    train_df = data_df.iloc[:train_size]
    val_df = data_df.iloc[train_size:train_size+val_size]
    test_df = data_df.iloc[train_size+val_size:]

    # Calculate Class Weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_df['regime']), y=train_df['regime'])
    class_weights = torch.FloatTensor(class_weights)
    print(f"Class Weights: {class_weights}")

    # Scale Features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df.drop(columns=['regime']))
    X_val = scaler.transform(val_df.drop(columns=['regime']))
    X_test = scaler.transform(test_df.drop(columns=['regime']))

    # Create Datasets
    target_ret_col = f'{config.target_ticker}_ret'
    y_train_reg = train_df[target_ret_col].values
    y_train_cls = train_df['regime'].values

    y_val_reg = val_df[target_ret_col].values
    y_val_cls = val_df['regime'].values

    train_ds = MarketDataset(X_train, y_train_reg, y_train_cls, config.lookback)
    val_ds = MarketDataset(X_val, y_val_reg, y_val_cls, config.lookback)
    test_ds = MarketDataset(X_test, test_df[target_ret_col].values, test_df['regime'].values, config.lookback)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    # 5. Initialize & Train
    print("Initializing model...")
    model = MarketRegimeTransformer(
        input_dim=X_train.shape[1],
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_layers=config.num_layers,
        num_regimes=config.num_regimes,
        dropout=config.dropout
    )

    print("Starting training...")
    history = train_model(model, train_loader, val_loader, config, class_weights, device)

    # Evaluation on Test Set
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    model.to(device)

    reg_preds = []
    cls_preds = []
    reg_true = []
    cls_true = []

    with torch.no_grad():
        for X, y_reg, y_cls in test_loader:
            X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
            out = model(X)
            reg_preds.append(out['reg'].cpu().numpy())
            cls_preds.append(torch.argmax(out['cls'], dim=1).cpu().numpy())
            reg_true.append(y_reg.cpu().numpy())
            cls_true.append(y_cls.cpu().numpy())

    reg_preds = np.concatenate(reg_preds)
    cls_preds = np.concatenate(cls_preds)
    reg_true = np.concatenate(reg_true)
    cls_true = np.concatenate(cls_true)

    # Metrics
    rmse = np.sqrt(mean_squared_error(reg_true, reg_preds))
    mae = mean_absolute_error(reg_true, reg_preds)
    acc = accuracy_score(cls_true, cls_preds)
    f1 = f1_score(cls_true, cls_preds, average='weighted')

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Regime Accuracy: {acc:.4f}")
    print(f"Regime F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(cls_true, cls_preds, target_names=['Bear', 'Neutral', 'Bull']))

if __name__ == "__main__":
    main()
