import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple

def download_ohlc(tickers: List[str], start: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.sort_index()
    df = df.dropna(how="all")
    return df

def build_features(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Target is ^GSPC log return next day
    prices = prices.ffill().bfill()
    logp = np.log(prices)
    rets = logp.diff()

    feats = []
    # raw returns
    for col in prices.columns:
        feats.append(rets[col].rename(f"ret_{col}"))
        # momentum
        feats.append(rets[col].rolling(5).mean().rename(f"mom5_{col}"))
        feats.append(rets[col].rolling(20).mean().rename(f"mom20_{col}"))
        # volatility
        feats.append(rets[col].rolling(20).std().rename(f"vol20_{col}"))
        # moving avg spreads
        ma20 = logp[col].rolling(20).mean()
        ma60 = logp[col].rolling(60).mean()
        feats.append((logp[col] - ma20).rename(f"dev20_{col}"))
        feats.append((ma20 - ma60).rename(f"ma20_60_{col}"))

    X = pd.concat(feats, axis=1).dropna()
    # Align target: next-day ret of ^GSPC
    y_reg = rets["^GSPC"].shift(-1).reindex(X.index)
    X = X.loc[y_reg.index].dropna()
    y_reg = y_reg.loc[X.index]
    return X, y_reg

def label_regimes(y_reg: pd.Series, bull: float = 0.015, bear: float = -0.015) -> pd.Series:
    # 20-day rolling return on ^GSPC closing price reconstructed from y_reg
    # Reconstruct price index (relative) from returns to compute rolling window sum
    # Using cumulative sum of daily returns approximation (log-returns)
    rets = y_reg.shift(1)  # avoid peeking at the target
    roll = rets.rolling(20).sum()
    lab = pd.Series(1, index=y_reg.index, dtype=np.int8)  # default neutral label
    lab[roll > bull] = 2   # bull
    lab[roll < bear] = 0   # bear
    lab[roll.isna()] = 1   # ensure early NaNs remain neutral
    return lab.astype(np.int8)

def train_val_test_split(df: pd.DataFrame, train_start: str, val_start: str, test_start: str):
    train = df.loc[df.index < val_start]
    val = df.loc[(df.index >= val_start) & (df.index < test_start)]
    test = df.loc[df.index >= test_start]
    return train, val, test

def zscore_fit(train_df: pd.DataFrame):
    mu = train_df.mean()
    sigma = train_df.std().replace(0, 1.0)
    return mu, sigma

def zscore_apply(df: pd.DataFrame, mu: pd.Series, sigma: pd.Series):
    return (df - mu) / sigma
