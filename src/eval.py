import argparse, os, numpy as np, pandas as pd, torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from .utils import device
from .data import download_ohlc, build_features, label_regimes, zscore_apply
from .dataset import WindowDataset
from .model import RegimeSwitchingTransformer

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tickers", nargs="+", default=["^GSPC","^VIX","^IXIC","GLD"])
    ap.add_argument("--start", type=str, default="2000-01-01")
    ap.add_argument("--val-start", type=str, default="2016-01-01")
    ap.add_argument("--test-start", type=str, default="2020-01-01")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--bull", type=float, default=0.015)
    ap.add_argument("--bear", type=float, default=-0.015)
    ap.add_argument("--batch-size", type=int, default=256)
    return ap.parse_args()

def main():
    args = parse_args()
    d = device()
    import pandas as pd
    import torch.serialization as ser
    ser.add_safe_globals([pd.Series])  # allow pandas Series
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mu, sigma, tr_args = ckpt["mu"], ckpt["sigma"], ckpt["args"]

    prices = download_ohlc(args.tickers, args.start)
    X, y_reg = build_features(prices)
    y_cls = label_regimes(y_reg, bull=args.bull, bear=args.bear)
    df = (X.join(y_reg.rename("y_reg")).join(y_cls.rename("y_cls"))).dropna()
    test_df = df.loc[df.index >= args.test_start]
    X_test = zscore_apply(test_df[X.columns], mu, sigma)

    ds_test = WindowDataset(X_test, test_df["y_reg"], test_df["y_cls"], window=args.window, horizon=args.horizon)
    loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    model = RegimeSwitchingTransformer(
        d_in=X.shape[1], d_model=tr_args["d_model"], nhead=tr_args["nhead"], num_layers=tr_args["layers"], dropout=tr_args["dropout"]
    ).to(d)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds_r, trues_r, preds_c, trues_c = [], [], [], []
    with torch.no_grad():
        for xb, yr, yc in loader:
            xb = xb.to(d)
            yhat_r, yhat_c = model(xb)
            preds_r.extend(yhat_r.squeeze().cpu().numpy().tolist())
            trues_r.extend(yr.squeeze().cpu().numpy().tolist())
            preds_c.extend(torch.argmax(yhat_c, dim=1).cpu().numpy().tolist())
            trues_c.extend(yc.cpu().numpy().tolist())

    mae = mean_absolute_error(trues_r, preds_r)
    rmse = mean_squared_error(trues_r, preds_r) ** 0.5
    dir_acc = np.mean((np.sign(trues_r) == np.sign(preds_r)).astype(float))
    acc = accuracy_score(trues_c, preds_c)
    f1  = f1_score(trues_c, preds_c, average="macro")
    cm  = confusion_matrix(trues_c, preds_c)

    print("=== Test Metrics ===")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Directional Accuracy: {dir_acc:.3f}")
    print(f"Regime ACC: {acc:.3f}  F1: {f1:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

if __name__ == "__main__":
    main()
