# Market Regime-Switching Transformer for Financial Time-Series Forecasting

**Course:** DATA 612 – Deep Learning  
**Team Members:** Juhi Ramod | Palak Wadhwa | Rohan Jain | Vikranth Reddimasu

---

## 1. Project Overview

This repository will host our work on building a **Market Regime-Switching Transformer (MRST)** — a deep learning model designed to forecast financial time-series data while **adapting to market regime changes** such as bull, bear, and neutral phases.

Unlike traditional models that assume a single stationary process, our approach aims to **jointly learn market return prediction and regime detection**, enabling more robust and interpretable forecasting during volatile market periods.

The project is currently in its **initial planning and research stage**. Model implementation and experiments will follow in upcoming development phases.

---

## 2. Problem Statement

Financial markets shift between regimes with different volatility structures and correlation dynamics. Most forecasting models treat these regime shifts as noise, leading to poor performance during transitions.

**Planned Solution**
- Develop a Transformer-based architecture with:
  - A **Regression Head** for predicting next-day market returns.
  - A **Classification Head** for identifying the current market regime.
- Incorporate macroeconomic and volatility indicators for enhanced context.
- Use rolling validation to evaluate model adaptability during different market cycles.

---

## 2.1 Design Decisions (Expert Defaults)

- Universe: `^GSPC` (target), `SPY`, `^IXIC` as covariates.
- Macros (FRED): `VIXCLS`, `FEDFUNDS`, `DGS10`, `CPIAUCSL`, `INDPRO`, `UNRATE`.
- Features: multi-scale momentum/volatility for each `close_*`, plus target log returns, RSI, SMA, and volatility.
- Labeling: 3-state HMM on returns+vol with majority smoothing (window=5); threshold labeler also available.
- Model: Transformer encoder with FiLM modulation and head gating; dual heads for regression and classification.
- Loss: MSE (or Quantile) + Cross-Entropy + optional correlation loss.
- Training: AMP, cosine LR with warmup, early stopping, grad clipping; DDP-ready for H100.
- Evaluation: RMSE/MAE/Directional, Acc/F1, Sharpe, MaxDD; simple backtest with 1bp cost.

---

## 3. Planned Data Sources

We plan to use publicly available datasets from:

- [Yahoo Finance](https://finance.yahoo.com) – S&P 500 (^GSPC), SPY ETF, Nasdaq Index  
- [FRED](https://fred.stlouisfed.org) – VIX, interest rates, CPI, industrial production  
- [Quandl](https://www.quandl.com) / Kaggle – Oil, gold, USD index  

**Key Features to Engineer**
- Log returns, volatility indicators, momentum signals  
- Macroeconomic factors (e.g., interest rates, CPI, VIX)  
- Regime labels derived using weak supervision (rolling return thresholds)

---

## 4. Planned Model Architecture

| Component                | Description                                                                 |
|---------------------------|------------------------------------------------------------------------------|
| Input                     | Multivariate time series (e.g., 60-day lookback)                             |
| Embedding Layer           | Linear projection + positional encoding                                     |
| Transformer Encoder       | Multi-head self-attention to model long-term dependencies                   |
| Output Heads              | (1) Regression (MSE loss)  (2) Classification (Cross-Entropy loss)          |
| Combined Loss             | `L = λ₁ × MSE + λ₂ × CrossEntropy`                                          |
| Framework                 | PyTorch + torchmetrics + scikit-learn + NumPy + Matplotlib                  |

---

## 5. Planned Evaluation

- **Forecasting**: MAE, RMSE, Directional Accuracy  
- **Regime Classification**: Accuracy, F1 Score, Confusion Matrix  
- **Interpretability**: Attention heatmaps, regime transition plots  
- **Baselines**: LSTM, Vanilla Transformer, Hidden Markov Models

---

## 6. Project Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Literature Review & Data Source Identification | In Progress |
| Phase 2 | Data Preprocessing & Feature Engineering | Planned |
| Phase 3 | Model Architecture Implementation | Planned |
| Phase 4 | Training & Evaluation | Planned |
| Phase 5 | Visualization, Interpretability, Report | Planned |

---

## 7. References

- Vaswani, A. et al. “Attention Is All You Need.” *NeurIPS*, 2017.  
- Lim, B. et al. “Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.” *NeurIPS*, 2019.  
- Wood, K. “Trading Momentum with Transformers.” GitHub Repository, 2022.  
- Pou, J. “Regimetry: Unsupervised Market Regime Detection Using Transformers.” GitHub Repository, 2023.  
- Rahimi, A. “Forecasting Economic and Market Regimes.” GitHub Repository, 2022.  
- FRED & Yahoo Finance data portals.

---

## 8. Repository Structure (Planned)

```

market-regime-transformer/
│── README.md
│── requirements.txt
│── notebooks/            # Exploratory analysis & model experiments
│── data/                 # Raw & processed datasets
│── src/                  # Core model & training code
│── utils/                # Data loaders, metrics, helpers
│── results/              # Visualizations, attention maps, forecasts
└── references/           # Papers, literature, resources

```

---

## 9. Getting Started (Coming Soon)

Quickstart for the first end-to-end baseline is now available.

1) Install dependencies

- Create and activate a Python 3.10+ env
- `pip install -r requirements.txt`

2) Prepare data (uses cached CSVs if available; otherwise downloads via yfinance)

- `python scripts/prepare_data.py`

3) Train MRST baseline

- `python scripts/train_mrst.py --labeling threshold --out results/experiments/mrst_threshold`
- Optional HMM labeling (requires `hmmlearn`):
  `python scripts/train_mrst.py --labeling hmm --out results/experiments/mrst_hmm`
 - Optional YAML config: `python scripts/train_mrst.py --config configs/mrst_default.yaml`

4) Results

- Check `results/experiments/.../history.json` and `results.json`
- Use plotting helpers in `src/visualization/plots.py`

5) Evaluate and backtest

- `python scripts/evaluate.py` (reads last run config and model from `results/experiments/mrst_threshold` by default)

6) Zaratan (H100) Slurm template

- Edit `scripts/slurm/train_mrst.slurm` to match your account/partition/modules
- Submit: `sbatch scripts/slurm/train_mrst.slurm`

Multi-GPU DDP (single node):

- Submit: `sbatch scripts/slurm/train_mrst_ddp.slurm`
- Uses `torchrun` and automatically enables DDP + AMP.

Walk-forward (rolling) training:

- `python scripts/train_walkforward.py`
- Saves per-fold results under `results/experiments/walkforward/` and a `summary.json`.

FRED API key (optional macros):

- Export: `export FRED_API_KEY=your_key_here` to fetch and cache macro series; otherwise, the pipeline runs with market data only.

---

## 10. License

This project will be open-sourced under the MIT License once implementation begins.
