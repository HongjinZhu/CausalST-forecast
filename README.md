# CausalST-forecast

A benchmarking framework for short-term solar power forecasting using meteorological data and deep learning models.  
This project combines climate correlation analysis (Granger causality, Pearson correlation) with predictive modeling (XGBoost, LSTM, and ST-ViT) to explore the impact of climate variables on solar energy output.

---

## Project Overview

This repository is part of a research project aiming to forecast solar field output from climate data using machine learning. Our focus is on:

- Building a reproducible **benchmark** for forecasting models (XGBoost, LSTM, ST-ViT)
- Using **ERA5 reanalysis data** as the primary feature source
- Integrating **climate variable interaction analysis** into model design
- Comparing forecasting architectures under the same input conditions

> Final goal: Evaluate whether advanced spatiotemporal models like ST-ViT provide meaningful improvements over traditional baselines under real-world climate conditions.

---

## Repository Structure

era5-solar-benchmark/
├── data/
│ ├── era5_features.csv
│ └── pv_output.csv
├── models/
│ ├── train_xgboost.py
│ ├── train_lstm.py
│ └── train_stvit.py
├── utils/
│ ├── preprocessing.py
│ ├── eval.py
├── benchmark_pipeline.ipynb
├── results/
│ ├── model_outputs/
│ └── evaluation_plots/
└── README.md


---

## Models Implemented

| Model    | Type        | Input                   | Description                          |
|----------|-------------|--------------------------|--------------------------------------|
| XGBoost  | Tabular ML  | ERA5 variables + time    | Simple but strong baseline           |
| LSTM     | Temporal DL | Sequences of ERA5 vars   | Captures time dependencies           |
| ST-ViT   | Vision DL   | Spatiotemporal features  | Transformer-based attention model    |

---

## Climate Analysis Module

We use Granger causality and Pearson correlation to:
- Explore lagged relationships between climate variables (e.g., humidity → cloud cover)
- Justify input feature selection and time window size for sequence models
- Provide interpretability and insight into model performance

---

## Getting Started

1. Clone the repo:
```bash
git clone https://github.com/your_username/era5-solar-benchmark.git
cd era5-solar-benchmark
```

2. Set up environment

```bash
pip install -r requirements.txt
```

4. Run benchmark:

```bash
python models/train_xgboost.py
python models/train_lstm.py
# python models/train_stvit.py (coming soon)
```

---

## Citation


---

## Contributors



---

## License


