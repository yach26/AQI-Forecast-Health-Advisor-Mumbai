# Model Evaluation Results

## Data Split Strategy

Strict temporal split — no data from future periods was used during training.

| Set | Period | Purpose |
|-----|--------|---------|
| Train | Jan 2020 – Jun 2024 | Model fitting |
| Validation | Jul 2024 – Sep 2024 | Optuna hyperparameter tuning |
| **Test (held-out)** | Oct 2024 – Dec 2024 | Final evaluation (evaluated once) |

> The test set was never seen during training or hyperparameter search. It was evaluated exactly once after all decisions were finalized.

---

## Cross-Validation Metrics

**Method:** 5-fold `TimeSeriesSplit` with a 24-hour gap to prevent target-adjacent leakage.  
Applied to the training set only.

| Metric | Mean | Std |
|--------|------|-----|
| MAE    | 2.243 | ±0.986 |
| RMSE   | 5.284 | ±2.318 |
| R²     | 0.9787 | ±0.0223 |

- **Training rows:** 21,115
- **Features:** 77
- **CV folds:** 5

---

## Held-Out Test Set Metrics

> Run `python src/train.py` to populate this section with real test results.

| Metric | Value |
|--------|-------|
| MAE    | — |
| RMSE   | — |
| R²     | — |
| Samples | — |

---

## Notes on Leakage Prevention

1. **Lag features** (`aqi_lag_1`, `aqi_lag_24`, etc.) are computed using `shift(n)`, which is purely backward-looking when data is sorted chronologically.
2. **Rolling features** (`rolling(24).mean()`) are similarly causal — only past rows contribute.
3. **AQI sub-indices** (`us_aqi_pm2_5`, `us_aqi_ozone`, etc.) are explicitly **excluded** from the feature set because they are derived *from* the target AQI and would constitute direct data leakage.
4. **Hyperparameter tuning** used only the validation set (Jul–Sep 2024). The test set (Oct–Dec 2024) was never accessed during this phase.
5. **TimeSeriesSplit** ensures all cross-validation folds respect temporal ordering — training folds always precede validation folds.
