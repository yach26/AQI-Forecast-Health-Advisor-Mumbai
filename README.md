# Mumbai AQI Forecast & AI Health Advisor

**An end-to-end machine learning system that forecasts Mumbai's Air Quality Index (AQI) up to 24 hours ahead and provides personalized AI-powered health recommendations — built with XGBoost, Streamlit, and the Groq Llama API.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange?logo=xgboost)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aqi-forecast-health-advisor-mumbai.streamlit.app/)

**[https://aqi-forecast-health-advisor-mumbai.streamlit.app/](https://aqi-forecast-health-advisor-mumbai.streamlit.app/)**

---

## My Journey Building This Project

### Why I Started

I live in Mumbai. During winter months the air sometimes burns your eyes just standing outside. I'd check apps, but they'd show a generic "Moderate" label and give no context for *what that means for me personally* or *what to expect tomorrow*. I wanted to build something that actually answered those questions.

This project started as a weekend experiment. It ended up taking half a month and teaching me more about ML engineering than anything else I've done.

### Version 1 — "It Works But I Have No Idea If It's Right"

The first version was a disaster in the best way.

I pulled 1 year of data from Open-Meteo, threw it into an XGBoost model with some lag features, got an R² of 0.62, and felt incredibly proud. I built a Streamlit dashboard. It looked okay.

Then I showed it to a friend who studies atmospheric science. He asked one question:

*"How did you handle the data split for your lag features?"*

I had no idea what he was talking about.

After some research, I realized I had computed rolling averages on the **entire dataset** and then shuffled it into train/test sets. My 24-hour lag for row 5000 was using row 4976's value — which was in the "validation" set I'd claimed was unseen. My 0.85 validation R² was completely fabricated by my own mistake.

I deleted the notebook and started over.

### What I Fixed in Version 2

**The real challenge:** Fix the leakage without destroying the model's ability to make predictions on live data.

The solution was separating *training-time feature engineering* from *inference-time feature engineering*:

- **Training:** Compute lags on the full chronologically-sorted dataset, then split by date. `shift(n)` in pandas is causal when data is sorted — it only looks backward.
- **Evaluation:** `TimeSeriesSplit(n_splits=5, gap=24)` — the 24-hour gap prevents the model from essentially predicting the immediate future using data from an hour ago in a different split.
- **Inference:** The live dashboard fetches 7 past days of data specifically to build a 72-hour lag history before making the current prediction.

I also expanded from 1 year → 5 years of data, added 40+ new features (boundary layer height, festival flags, dew point, UV index), and ran 100 Optuna trials to tune hyperparameters.

The result: **R² went from 0.62 → 0.979**.

### Technical Challenges & How I Solved Them

**Challenge 1: The pandas `read_json` breaking change**

When Streamlit's caching serialized a DataFrame to JSON for the 7-day comparison tab, the newer version of pandas refused to parse a raw JSON string — it treated it as a file path. Fixed by wrapping with `io.StringIO()`.

**Challenge 2: Feature alignment at inference time**

The training set had 77 features including wind direction dummies (`wdir_NE`, `wdir_SW`, etc.) that are only created when those wind directions actually appear in data. If the live API returns wind data with directions not seen in the 7-day window, those dummy columns are missing.

Solution: At inference time, iterate through the training feature list and fill any missing columns with `0.0` before passing to the model.

**Challenge 3: Honest 7-day forecasting**

True recursive forecasting (predict t+24 → use as lag → predict t+48 → ...) requires all future weather inputs, which Open-Meteo provides for 7 days. But the model expects ~77 features including rolling AQI statistics — and future AQI is what we're trying to predict.

I implemented a seasonal mean-reversion simulation for the "Model" column in the 7-day tab, and clearly labeled it as a scenario-based outlook rather than a true recursive forecast. The honest labeling matters more than faking accuracy.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Data Layer                              │
│  Open-Meteo Air Quality API  ←→  Open-Meteo Weather API   │
│  (Real-time + 7-day past history)                          │
└───────────────────────┬────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────┐
│                Feature Engineering                         │
│  Lag features (1h→72h)  |  Rolling windows (3h→48h)       │
│  Cyclical encodings      |  Season flags (Mumbai IMD)      │
│  Festival dates          |  Interaction features           │
└───────────────────────┬────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────┐
│              XGBoost Model (v2)                            │
│  5-year training data  |  100 Optuna trials                │
│  TimeSeriesSplit CV    |  77 features                      │
└─────────────┬──────────────────┬──────────────────────────┘
              │                  │
┌─────────────▼──────┐   ┌──────▼──────────────────────────┐
│   AQI Prediction   │   │   SHAP Explainability           │
│   (24h ahead)      │   │   (TreeExplainer)               │
└─────────────┬──────┘   └──────────────────────────────────┘
              │
┌─────────────▼──────────────────────────────────────────────┐
│              Groq Llama-3.3-70b                            │
│  Personalized health recommendations based on AQI,         │
│  user profile (age, conditions), and current pollutants    │
└─────────────┬──────────────────────────────────────────────┘
              │
┌─────────────▼──────────────────────────────────────────────┐
│           Streamlit Dashboard                              │
│  Live AQI gauge | 48h history | 7-day outlook | SHAP      │
│  Health advisor | Weather metrics | Model insights         │
└────────────────────────────────────────────────────────────┘
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Open-Meteo Historical Air Quality + Weather Archive APIs |
| Location | Mumbai, Maharashtra (19.0760°N, 72.8777°E) |
| Period | Jan 2020 – Dec 2024 (5 years) |
| Frequency | Hourly |
| Rows | ~43,800 |
| Size | ~15 MB |

### Reproducibility

The dataset is not committed (large file). Regenerate it with one command:

```bash
python src/build_dataset.py
```

---

## Feature Engineering

77 features across 8 groups:

### Temporal (7 features)
| Feature | Reasoning |
|---------|-----------|
| `hour`, `day_of_week`, `month` | AQI has strong diurnal and weekly patterns (rush hour, weekends) |
| `week_of_year`, `day_of_year` | Seasonal gradients within a year |
| `is_weekend` | Reduced industrial + traffic emissions on weekends |

### Cyclical Encodings (6 features)
`hour_sin/cos`, `month_sin/cos`, `dow_sin/cos` — encodes periodicity without creating a discontinuity at the boundary (e.g., hour 23 → hour 0).

### Lag Features (20 features)
| Feature | Reasoning |
|---------|-----------|
| `aqi_lag_1` to `aqi_lag_72` | AQI has autocorrelation ~0.97 at lag 1 (SHAP shows these as top predictors) |
| `pm25_lag_1/3/6/24` | PM2.5 is the dominant AQI component in Mumbai |
| `precip_lag_1..24` | Rainfall washout effect is delayed — precipitation 6–12h ago matters |
| `wind_lag_1/6` | Wind speed changes affect dispersion with a delay |

### Rolling Statistics (12 features)
3h, 6h, 12h, 24h, 48h rolling means/std/max/min — captures pollution persistence and trend.

### Mumbai Season Flags (3 features)
Based on **India Meteorological Department (IMD)** seasonal classification for the Maharashtra coast:
- **Monsoon (Jun–Sep):** Heavy rainfall + SW Arabian Sea winds → lowest AQI. Reference: IMD Seasonal Climatology for Maharashtra.
- **Post-monsoon (Oct–Nov):** Transition. Pollution starts rising.
- **Winter (Dec–Feb):** Temperature inversions trap pollutants. Worst AQI period.
- **Summer (Mar–May):** Hot, dusty. Reference category (dropped from dummies).

### Festival Flags (1 feature)
Mumbai-specific pollution events with historically documented PM2.5/PM10 spikes:
- **Diwali:** Fireworks → PM2.5 3–5× baseline on festival night and morning after
- **Holi:** Bonfires → CO and black carbon spikes
- **Ganesh Chaturthi:** Firecrackers + heavy vehicle traffic during Visarjan processions

Flagged on the festival day AND the following day (pollution lingers 12–24h).

### Interaction Features (3 features)
| Feature | Formula | Reasoning |
|---------|---------|-----------|
| `ventilation_idx` | BLH × wind_speed / 1000 | Higher = pollutants disperse faster |
| `inversion_proxy` | BLH < 500m | Temperature inversion flag — trapping layer |
| `humidity_pm25` | humidity × PM2.5 / 100 | Hygroscopic growth: PM2.5 swells in humid air, increasing optical depth and AQI |

**Boundary Layer Height (BLH):** Provided by Open-Meteo's atmospheric boundary layer model. A BLH < 500m indicates a capping inversion layer — a meteorological condition strongly associated with Mumbai's worst winter pollution episodes.

---

## Model Training

### Pipeline

```
Raw Data (5yr, 43,800 rows)
         │
         ▼
Feature Engineering  ← Computed on sorted full dataset (causal)
         │
    ┌────┴──────────────────────────────────────────────────┐
    │                  │                   │                │
  Train             Val               (never seen)       Test
Jan 2020–         Jul 2024–            ↑                Oct 2024–
Jun 2024          Sep 2024          Leakage             Dec 2024
  (~80%)           (~10%)          firewall              (~10%)
    │                  │
    ▼                  │
5-fold TimeSeriesSplit │
  CV on Train only     │
    │                  │
    ▼                  ▼
Optuna HPO (100 trials, minimizing val RMSE)
    │
    ▼
Final Model trained on Train + Val
    │
    ▼
Evaluated ONCE on Test set
```

### Cross-Validation

`TimeSeriesSplit(n_splits=5, gap=24)` — the 24-hour gap ensures that the 24h-ahead lag features in the validation set cannot peek at the training set's final hours.

### Hyperparameter Optimization

Optuna `TPESampler` with 100 trials. Search space:

| Parameter | Range |
|-----------|-------|
| `n_estimators` | 300–1200 |
| `max_depth` | 3–9 |
| `learning_rate` | 0.01–0.15 (log scale) |
| `subsample` | 0.6–1.0 |
| `colsample_bytree` | 0.5–1.0 |
| `min_child_weight` | 1–10 |
| `gamma` | 0.0–1.0 |
| `reg_alpha` | 0.0–2.0 |
| `reg_lambda` | 0.5–3.0 |

Best params found: `n_estimators=800, max_depth=7, learning_rate=0.044`

---

## Results

### Cross-Validation (5-fold TimeSeriesSplit)

| Metric | Mean | Std |
|--------|------|-----|
| **MAE** | **2.24** | ±0.99 |
| **RMSE** | **5.28** | ±2.32 |
| **R²** | **0.979** | ±0.022 |

Training rows: 21,115 | Features: 77

### Held-Out Test Set (Oct–Dec 2024)

Evaluated exactly once after all decisions were finalized.

| Metric | Value |
|--------|-------|
| **MAE** | **1.431** |
| **RMSE** | **2.734** |
| **R²** | **0.9964** |
| Samples | 2,208 |

---

## 🚨 The Truth About The 0.99 R² (Why It's Still "Wrong")

You might look at an R² of 0.996 and think this is the greatest weather model ever built. **It is not. It is actually answering the wrong question.**

While I fixed the *cross-validation* leakage (using `TimeSeriesSplit`), there is still a fundamental horizon mismatch in the *features*. 

The project claims to "forecast AQI up to 24 hours ahead". However, the most important features in the model are `aqi_lag_1`, `aqi_roll_mean_3`, etc. 

If you are trying to predict the AQI for tomorrow at 5 PM (24 hours from now), you **do not have** the AQI for tomorrow at 4 PM (`aqi_lag_1`). The most recent data you have is from *right now* (`aqi_lag_24`). 

Because the model was trained using `aqi_lag_1` to predict the target `us_aqi`, it is effectively a **1-hour ahead forecasting model**, not a 24-hour ahead model. It achieves near-perfect accuracy because the air quality 1 hour from now is almost exactly the same as the air quality right now. 

**To build a true 24-hour ahead model, I would need to:**
1. Drop all lags smaller than 24 hours (`lag_1` through `lag_23`).
2. Only use weather forecasts (not historical weather) for the future 24-hour window.
3. Or, build an autoregressive / recursive model that predicts `t+1`, feeds that prediction back in as the new `lag_1`, predicts `t+2`, and so on up to `t+24` (where errors compound rapidly).

The current model is highly accurate for real-time nowcasting and 1-hour look-aheads, but the 24-hour dashboard simulation is using a mean-reversion trick because the XGBoost model mathematically cannot forecast 24 hours ahead without future lag data.

---

## SHAP Explainability

SHAP TreeExplainer was run on a 2,000-row sample. Top features by mean |SHAP|:

1. `aqi_lag_1` — 1-hour AQI autocorrelation (strongest predictor)
2. `aqi_lag_24` — 24-hour pattern (same time yesterday)
3. `aqi_roll_mean_24` — 24-hour moving average
4. `aqi_lag_3`, `aqi_lag_6` — short-term momentum
5. `boundary_layer_height` — atmospheric trapping layer
6. `pm2_5` — current fine particulate matter
7. `ventilation_idx` — dispersion capacity
8. `humidity_pm25` — humidity interaction
9. `hour_sin/cos` — time of day
10. `inversion_proxy` — temperature inversion flag

Full SHAP summary at `notebooks/model/shap_summary.csv`.

---

## Streamlit Dashboard Features

| Tab | What It Shows |
|-----|--------------|
| **Live AQI** | Real-time AQI gauge, 24h forecast with confidence interval, current pollutant concentrations |
| **48h History** | Interactive Plotly chart of the last 48 hours |
| **Health** | Personalized health guidance based on AQI level |
| **AI Advisor** | Groq Llama-3.3-70b generates custom recommendations based on user age/conditions |
| **Model Insights** | Feature importance chart, performance benchmarks, engineering details |
| **7-Day Outlook** | Scenario-based 7-day AQI outlook vs. Open-Meteo official forecast |

---

## Forecast Methodology

The 7-day "Model" forecast column uses a **seasonal mean-reversion simulation**, not true recursive XGBoost forecasting. It is labeled explicitly as a "Scenario Outlook" in the dashboard.

True recursive forecasting (predict t+24 → feed as lag → predict t+48 → ...) is on the roadmap but requires future weather inputs from Open-Meteo and careful handling of compounding prediction error.

---

## Future Improvements

- [ ] Implement true recursive multi-step XGBoost forecasting
- [ ] Add CPCB (Central Pollution Control Board) ground station validation data
- [ ] Experiment with LSTM or Temporal Fusion Transformer for long-range forecasting
- [ ] Add Dask/RAPIDS for faster feature engineering on larger datasets
- [ ] Deploy with GitHub Actions CI/CD pipeline
- [ ] Add Prometheus metrics for model drift detection

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yach26/AQI-Forecast-Health-Advisor-Mumbai.git
cd AQI-Forecast-Health-Advisor-Mumbai

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add your Groq API key for the AI Advisor tab
#    Create .streamlit/secrets.toml and add:
#    GROQ_API_KEY = "your_key_here"

# 5. Run the dashboard
streamlit run app.py
```

### Rebuild from Scratch

```bash
# Rebuild the dataset (requires internet, ~5 minutes)
python src/build_dataset.py

# Retrain the model (requires dataset, ~30–60 minutes with Optuna)
python src/train.py

# Quick test run (no Optuna, uses default params, ~2 minutes)
python src/train.py --dry-run --trials 10

# Evaluate the saved model on the test set
python src/evaluate.py

# Run tests
python -m pytest tests/ -v
```

---

## Repository Structure

```
AQI-Forecast-Health-Advisor-Mumbai/
├── app.py                          # Streamlit dashboard entry point
├── requirements.txt                # Python dependencies
├── LICENSE
│
├── src/                            # Production source code
│   ├── __init__.py
│   ├── build_dataset.py            # Fetch + engineer 5yr dataset
│   ├── train.py                    # Full training pipeline
│   ├── evaluate.py                 # Standalone evaluation
│   ├── core_utils.py               # Live inference + feature engineering
│   ├── llm_utils.py                # Groq AI health advisor
│   └── forecast_compare_utils.py   # 7-day scenario outlook
│
├── notebooks/                      # Jupyter notebooks (exploration)
│   ├── 01_build_dataset_v2.ipynb   # Interactive dataset builder
│   ├── 02_train_model_v2.ipynb     # Interactive training + SHAP
│   ├── backup/                     # Archived experiments
│   └── model/                      # Model artifacts
│       ├── xgb_model_v2.pkl        # Trained XGBoost model (4.1 MB)
│       ├── feature_cols_v2.pkl     # Feature column list
│       ├── feature_importance.csv  # XGBoost gain scores
│       ├── shap_summary.csv        # Mean |SHAP| per feature
│       ├── best_params.json        # Optuna best hyperparameters
│       └── cv_summary.json         # Cross-validation metrics
│
├── reports/                        # Evaluation outputs
│   ├── final_metrics.md            # Human-readable metrics report
│   └── test_results.json           # Machine-readable test metrics
│
├── assets/                         # Static files
│   └── Background.jpg
│
├── tests/                          # Unit tests
│   ├── test_features.py            # Lag/rolling leakage tests
│   └── test_inference.py           # Model output schema tests
│
└── data/                           # (gitignored — regenerate with build_dataset.py)
    └── mumbai_aqi_5yr_expanded.csv
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Open-Meteo Historical Air Quality & Weather APIs |
| ML | XGBoost 2.x, scikit-learn, Optuna, SHAP |
| Dashboard | Streamlit, Plotly |
| AI Advisor | Groq (Llama-3.3-70b-versatile) |
| Language | Python 3.10+ |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with curiosity about the air we breathe in Mumbai.*
