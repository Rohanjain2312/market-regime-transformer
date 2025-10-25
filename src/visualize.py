import torch, pandas as pd, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .data import download_ohlc, build_features, label_regimes, zscore_apply
from .dataset import WindowDataset
from .model import RegimeSwitchingTransformer
from .utils import device
import torch.serialization as ser


def load_model(ckpt_path, tickers, start, test_start, window=60, horizon=1, bull=0.015, bear=-0.015):
    ser.add_safe_globals([pd.Series])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mu, sigma, tr_args = ckpt["mu"], ckpt["sigma"], ckpt["args"]

    prices = download_ohlc(tickers, start)
    X, y_reg = build_features(prices)
    y_cls = label_regimes(y_reg, bull=bull, bear=bear)
    df = (X.join(y_reg.rename("y_reg")).join(y_cls.rename("y_cls"))).dropna()
    test_df = df.loc[df.index >= test_start]
    X_test = zscore_apply(test_df[X.columns], mu, sigma)

    ds_test = WindowDataset(X_test, test_df["y_reg"], test_df["y_cls"], window=window, horizon=horizon)
    loader = DataLoader(ds_test, batch_size=1, shuffle=False)

    model = RegimeSwitchingTransformer(
        d_in=X.shape[1],
        d_model=tr_args["d_model"],
        nhead=tr_args["nhead"],
        num_layers=tr_args["layers"],
        dropout=tr_args["dropout"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, loader, test_df


def visualize(ckpt_path="outputs/best.pt", tickers=["^GSPC", "^VIX", "^IXIC", "GLD"],
              start="2000-01-01", test_start="2020-01-01"):
    model, loader, df = load_model(ckpt_path, tickers, start, test_start)
    d = device()
    preds_r, preds_c = [], []
    with torch.no_grad():
        for xb, _, _ in loader:
            xb = xb.to(d)
            yhat_r, yhat_c = model(xb)
            preds_r.append(yhat_r.item())
            preds_c.append(torch.argmax(yhat_c, dim=1).item())

    # Align and attach predictions
    df = df.loc[df.index[-len(preds_r):]]
    df["pred_return"] = preds_r
    df["pred_regime"] = preds_c

    # Map regime labels for readability
    regime_map = {0: "Bear", 1: "Neutral", 2: "Bull"}
    df["true_regime_name"] = df["y_cls"].map(regime_map)
    df["pred_regime_name"] = df["pred_regime"].map(regime_map)

    # --- Option 3: Two stacked subplots (Regime timeline + Returns) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2]})

    # --- Top plot: Regime timeline ---
    colors = {0: "blue", 1: "gray", 2: "red"}
    for regime, color in colors.items():
        mask = df["pred_regime"] == regime
        axes[0].fill_between(df.index, 0, 1, where=mask, color=color, alpha=0.4)

    axes[0].set_yticks([])
    axes[0].set_ylabel("Predicted Regime", fontsize=10)
    axes[0].set_title("Predicted Market Regimes (2020–2024)")
    axes[0].legend(handles=[
        plt.Line2D([0], [0], color="red", lw=6, alpha=0.4, label="Bull"),
        plt.Line2D([0], [0], color="gray", lw=6, alpha=0.4, label="Neutral"),
        plt.Line2D([0], [0], color="blue", lw=6, alpha=0.4, label="Bear"),
    ], loc="upper right")

    # --- Bottom plot: Predicted vs Actual Returns ---
    axes[1].plot(df.index, df["y_reg"], label="Actual Return", color="black", linewidth=0.6)
    axes[1].plot(df.index, df["pred_return"], label="Predicted Return", color="red", alpha=0.7)
    axes[1].set_ylabel("Daily Return")
    axes[1].legend()
    axes[1].set_title("Predicted vs Actual Returns (2020–2024)")

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize()