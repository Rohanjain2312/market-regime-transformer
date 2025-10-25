import argparse, os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import set_seed, ensure_dir, device
from .data import download_ohlc, build_features, label_regimes, train_val_test_split, zscore_fit, zscore_apply
from .dataset import WindowDataset
from .model import RegimeSwitchingTransformer

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["^GSPC","^VIX","^IXIC","GLD"])
    ap.add_argument("--start", type=str, default="2000-01-01")
    ap.add_argument("--val-start", type=str, default="2016-01-01")
    ap.add_argument("--test-start", type=str, default="2020-01-01")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--bull", type=float, default=0.015)
    ap.add_argument("--bear", type=float, default=-0.015)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lambda-reg", type=float, default=1.0)
    ap.add_argument("--lambda-cls", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="outputs")
    return ap.parse_args()

def batchify(loader, d):
    for xb, yr, yc in loader:
        yield xb.to(d), yr.to(d), yc.to(d)

def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.outdir)
    d = device()

    # Data
    prices = download_ohlc(args.tickers, args.start)
    X, y_reg = build_features(prices)
    y_cls = label_regimes(y_reg, bull=args.bull, bear=args.bear)
    df = pd.concat([X, y_reg.rename("y_reg"), y_cls.rename("y_cls")], axis=1).dropna()

    train_df, val_df, test_df = train_val_test_split(df, args.start, args.val_start, args.test_start)
    mu, sigma = zscore_fit(train_df[X.columns])
    X_train = zscore_apply(train_df[X.columns], mu, sigma)
    X_val   = zscore_apply(val_df[X.columns], mu, sigma)
    X_test  = zscore_apply(test_df[X.columns], mu, sigma)

    ds_train = WindowDataset(X_train, train_df["y_reg"], train_df["y_cls"], window=args.window, horizon=args.horizon)
    ds_val   = WindowDataset(X_val,   val_df["y_reg"],   val_df["y_cls"],   window=args.window, horizon=args.horizon)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False)

    # Model
    model = RegimeSwitchingTransformer(
        d_in=X.shape[1], d_model=args.d_model, nhead=args.nhead, num_layers=args.layers, dropout=args.dropout
    ).to(d)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mse = nn.MSELoss()
    ce  = nn.CrossEntropyLoss()

    best_val = np.inf
    ckpt_path = os.path.join(args.outdir, "best.pt")

    for epoch in range(1, args.epochs+1):
        model.train()
        tr_losses = []
        for xb, yr, yc in tqdm(batchify(train_loader, d), total=len(train_loader), desc=f"Epoch {epoch}"):
            opt.zero_grad()
            yhat_r, yhat_c = model(xb)
            loss = args.lambda_reg*mse(yhat_r.squeeze(), yr.squeeze()) + args.lambda_cls*ce(yhat_c, yc)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())
        sched.step()

        # val
        model.eval()
        with torch.no_grad():
            v_losses = []
            preds_r, trues_r, preds_c, trues_c = [], [], [], []
            for xb, yr, yc in batchify(val_loader, d):
                yhat_r, yhat_c = model(xb)
                v_loss = args.lambda_reg*mse(yhat_r.squeeze(), yr.squeeze()) + args.lambda_cls*ce(yhat_c, yc)
                v_losses.append(v_loss.item())
                preds_r.extend(yhat_r.squeeze().cpu().numpy().tolist())
                trues_r.extend(yr.squeeze().cpu().numpy().tolist())
                preds_c.extend(torch.argmax(yhat_c, dim=1).cpu().numpy().tolist())
                trues_c.extend(yc.cpu().numpy().tolist())
            mae = mean_absolute_error(trues_r, preds_r)
            rmse = mean_squared_error(trues_r, preds_r) ** 0.5
            acc = accuracy_score(trues_c, preds_c)
            f1  = f1_score(trues_c, preds_c, average="macro")
            val_score = mae + rmse  # simple composite
        print(f"[Epoch {epoch}] train_loss={np.mean(tr_losses):.4f} val_loss={np.mean(v_losses):.4f} MAE={mae:.4f} RMSE={rmse:.4f} ACC={acc:.3f} F1={f1:.3f}")

        if val_score < best_val:
            best_val = val_score
            torch.save({"model": model.state_dict(), "mu": mu, "sigma": sigma, "args": vars(args)}, ckpt_path)
            print(f"Saved checkpoint -> {ckpt_path}")

    print("Training done.")

if __name__ == "__main__":
    main()
