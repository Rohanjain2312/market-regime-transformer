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

This section will be updated once the first version of the codebase is ready.

---

## 10. License

This project will be open-sourced under the MIT License once implementation begins.


## Setup
```bash
pip install -r requirements.txt

Train

python -m src.train --tickers ^GSPC ^VIX ^IXIC GLD --start 2000-01-01 --val-start 2016-01-01 --test-start 2020-01-01 \
  --window 60 --horizon 1 --bull 0.015 --bear -0.015 --epochs 20 --batch-size 128 --d-model 256 --nhead 4 --layers 3

Evaluate

python -m src.eval --ckpt outputs/best.pt --tickers ^GSPC ^VIX ^IXIC GLD --start 2000-01-01 --val-start 2016-01-01 \
  --test-start 2020-01-01 --window 60 --horizon 1 --bull 0.015 --bear -0.015

Artifacts are saved to outputs/ (checkpoints, metrics).

---
