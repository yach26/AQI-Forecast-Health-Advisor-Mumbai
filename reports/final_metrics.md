# Model Evaluation Results

## Data Split

| Set | Period | Rows |
|-----|--------|------|
| Train | Jan 2020 – Jun 2024 | ~17664 |
| Validation | Jul 2024 – Sep 2024 | ~4416 |
| **Test (held-out)** | Oct 2024 – Dec 2024 | **2208** |

## Cross-Validation Metrics (5-fold TimeSeriesSplit, gap=24h)

| Metric | Mean | Std |
|--------|------|-----|
| MAE    | 2.279  | ±1.087 |
| RMSE   | 5.480 | ±2.739 |
| R²     | 0.9818 | ±0.0186 |

## Held-Out Test Set Metrics

| Metric | Value |
|--------|-------|
| MAE    | 1.431  |
| RMSE   | 2.734 |
| R²     | 0.9964 |

> Test set was evaluated exactly once, after all hyperparameter decisions were finalized.
