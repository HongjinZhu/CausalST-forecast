# CausalST-forecast

A benchmarking and analysis framework for short-term solar power forecasting using meteorological reanalysis data, satellite-based spatial features, and deep learning models.
This project integrates **climate variable interaction analysis** (Granger causality, Pearson correlation) with **predictive modeling** (XGBoost, LSTM, and ST-ViT) to study and improve the impact of climate and spatial datasets on solar energy output forecasting.

---

## Project Overview

This repository is part of a research project aiming to forecast solar field output from climate and remote sensing data using machine learning.
Our focus is on:

* Building a reproducible **benchmark** for forecasting models (XGBoost, LSTM, ST-ViT)
* Using **ERA5 reanalysis** as the backbone meteorological feature set
* Integrating **satellite-derived spatial datasets** (Sentinel-2, Sentinel-1, HLS, Landsat 8, MODIS, VIIRS) matched to the ERA5 hourly grid
* Applying **gap-aware nearest-hour and interpolation strategies** for sparse datasets
* Using **correlation and causality analysis** to drive feature and time-window selection
* Comparing forecasting architectures under consistent input conditions

> **Final goal:** Evaluate whether advanced spatiotemporal models like ST-ViT provide meaningful improvements over strong baselines under real-world climate and data-availability constraints.

---

## Repository Structure

```
causalst-forecast/
├── data/
│   └── README.md
├── models/
│   ├── train_xgboost.py               # Baseline tabular model
│   ├── train_lstm.py                  # Sequence model
│   └── train_stvit.py                 # Spatiotemporal Vision Transformer
├── spatiotemporal_match/
│   ├── match_utils.py                 # Matching/interpolation helpers
│   ├── stats_utils.py                 # Correlation & Granger causality tools
│   ├── spatiotemporal_utils.py        # Provenance features & gap caps
│   └── match_test.ipynb
├── utils/
│   ├── download_pressure-level_data.py
│   ├── download_single-level_data.py
│   ├── preprocessing.py
│   └── eval.py
├── benchmark.ipynb
├── ST-ViT.ipynb
├── requirements.txt
└── README.md
```

---

## Data Matching & Interpolation

We align sparse satellite observations to the ERA5 hourly grid using **`match_utils.py`**:

* **Nearest-hour matching** for categorical/regime features (e.g., cloud mask, SAR backscatter)
* **Linear and spline interpolation** (gap-aware) for continuous features (e.g., NDVI, cloud score)
* **Provenance features**:

  * `is_anchor` (1 if a real observation)
  * `time_since_last_obs` / `gap_to_next_obs`
  * `valid_frac` (fraction of pixels passing QA)
* **Gap caps**: Avoid extrapolating across long gaps

Datasets in scope:

* Sentinel-2 L2A Cloud Score+
* Sentinel-1 GRD (VV/VH)
* HLSL30 NDVI
* Landsat 8 SR
* MODIS MOD09GA & Cloud Mask
* VIIRS Cloud Mask

---

## Climate Analysis Module

We use **`stats_utils.py`** for:

* **Pearson correlation** to identify linear associations
* **Granger causality** to test whether variable A improves forecasts of variable B at specific lags
* Lagged-relationship maps to decide:

  * Which variables to include
  * How far back in time the model should “look” for predictive signal

---

## Getting Started

1. Clone the repo:

```bash
git clone https://github.com/your_username/causalst-forecast.git
cd causalst-forecast
```

2. Set up environment:

```bash
pip install -r requirements.txt
```

3. Prepare data:

* Download ERA5 features (hourly)
* Collect satellite datasets from Google Earth Engine or archives
* Use `match_utils.py` + `spatiotemporal_utils.py` to align features to ERA5 time grid

4. Run benchmark (improvement on-going):

```bash
python models/train_xgboost.py
python models/train_lstm.py
python models/train_stvit.py
```

5. Run climate analysis:

```python
from stats_utils import corr_matrix, granger_batch
corr = corr_matrix(out, cols=[...], time_col="time")
gc = granger_batch(out, pairs=[(..., ...)], time_col="time")
```

---

## Citation

*To be added*

---

## License

