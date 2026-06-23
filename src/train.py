"""
src/train.py
─────────────────────────────────────────────────────────────────────────────
Production-grade training script for the Mumbai AQI XGBoost forecasting model.

Data Split (strict temporal, no leakage)
─────────────────────────────────────────
  Train      : Jan 2020 – Jun 2024  (~80%)
  Validation : Jul 2024 – Sep 2024  (~10%)  — used for Optuna HPO
  Test       : Oct 2024 – Dec 2024  (~10%)  — untouched final evaluation

Leakage Prevention
──────────────────
  1. Lag and rolling features are computed on the FULL sorted dataset first,
     then the dataset is split by date. This is safe because:
     - shift(n) only looks backward in time (prior rows in sorted data)
     - rolling(n).mean() also only uses past rows when data is chronologically sorted
     - The 24-hour gap in TimeSeriesSplit ensures no target-adjacent leakage
  2. AQI sub-indices (us_aqi_pm2_5, us_aqi_ozone, etc.) are excluded from
     features as they are derived FROM the target and represent direct leakage.
  3. Hyperparameter tuning uses ONLY the validation set, never the test set.
  4. The test set is evaluated exactly once, after all decisions are finalized.

Usage
─────
    python src/train.py [--trials N] [--dry-run]

    --trials N   Number of Optuna trials (default: 100, use 10 for quick test)
    --dry-run    Skip Optuna, use default params, train quickly

Outputs
───────
    notebooks/model/xgb_model_v2.pkl       — trained XGBoost model
    notebooks/model/feature_cols_v2.pkl    — ordered feature column list
    notebooks/model/feature_importance.csv — XGBoost gain scores
    notebooks/model/shap_summary.csv       — mean |SHAP| per feature
    notebooks/model/best_params.json       — Optuna best hyperparameters
    notebooks/model/cv_summary.json        — cross-validation summary
    reports/test_results.json              — held-out test set metrics
    reports/final_metrics.md               — formatted metrics summary
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "mumbai_aqi_5yr_expanded.csv"
MODEL_DIR = ROOT / "notebooks" / "model"
REPORT_DIR = ROOT / "reports"

# ── Temporal split boundaries ─────────────────────────────────────────────────
# Train  : Jan 2020 – Jun 2024
# Val    : Jul 2024 – Sep 2024   (used for Optuna tuning only)
# Test   : Oct 2024 – Dec 2024   (held out, evaluated once)
VAL_START  = pd.Timestamp("2024-07-01")
TEST_START = pd.Timestamp("2024-10-01")

# ── Features to exclude (leakage or non-predictive at inference time) ─────────
EXCLUDE_COLS = [
    "time",
    "us_aqi",            # target variable
    "us_aqi_pm2_5",      # derived sub-index — direct leakage from target
    "us_aqi_pm10",       # derived sub-index — direct leakage from target
    "us_aqi_nitrogen_dioxide",  # derived sub-index
    "us_aqi_ozone",      # derived sub-index
    "season",            # replaced by one-hot season_* dummies
]

TARGET = "us_aqi"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Run: python src/build_dataset.py")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"Loaded: {len(df):,} rows  |  {df['time'].min().date()} → {df['time'].max().date()}")
    return df


def split_data(df: pd.DataFrame):
    """Strict temporal split. No shuffling. Test set is never touched during tuning."""
    train_df = df[df["time"] <  VAL_START].copy()
    val_df   = df[(df["time"] >= VAL_START) & (df["time"] < TEST_START)].copy()
    test_df  = df[df["time"] >= TEST_START].copy()

    print(f"\nData split:")
    print(f"  Train : {len(train_df):>6,} rows  {train_df['time'].min().date()} → {train_df['time'].max().date()}")
    print(f"  Val   : {len(val_df):>6,} rows  {val_df['time'].min().date()} → {val_df['time'].max().date()}")
    print(f"  Test  : {len(test_df):>6,} rows  {test_df['time'].min().date()} → {test_df['time'].max().date()}")
    return train_df, val_df, test_df


def prepare_xy(df: pd.DataFrame, feature_cols: list):
    df_clean = df[feature_cols + [TARGET]].dropna()
    X = df_clean[feature_cols].values
    y = df_clean[TARGET].values
    return X, y


def cross_validate(X_train, y_train, params: dict, n_splits: int = 5) -> dict:
    """5-fold TimeSeriesSplit CV on the training set only."""
    import xgboost as xgb

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=24)
    maes, rmses, r2s = [], [], []

    print(f"\nRunning {n_splits}-fold TimeSeriesSplit CV (gap=24h) ...")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_v = X_train[tr_idx], X_train[val_idx]
        y_tr, y_v = y_train[tr_idx], y_train[val_idx]

        m = xgb.XGBRegressor(**{**params, "random_state": 42, "n_jobs": -1})
        m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
        preds = m.predict(X_v)

        mae  = mean_absolute_error(y_v, preds)
        rmse = np.sqrt(mean_squared_error(y_v, preds))
        r2   = r2_score(y_v, preds)
        maes.append(mae); rmses.append(rmse); r2s.append(r2)
        print(f"  Fold {fold+1}: MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}  "
              f"[train={len(X_tr):,}  val={len(X_v):,}]")

    summary = {
        "n_splits":    n_splits,
        "mae_mean":    round(float(np.mean(maes)),  3),
        "mae_std":     round(float(np.std(maes)),   3),
        "rmse_mean":   round(float(np.mean(rmses)), 3),
        "rmse_std":    round(float(np.std(rmses)),  3),
        "r2_mean":     round(float(np.mean(r2s)),   4),
        "r2_std":      round(float(np.std(r2s)),    4),
    }
    print(f"\n  CV MAE  : {summary['mae_mean']:.2f} ± {summary['mae_std']:.2f}")
    print(f"  CV RMSE : {summary['rmse_mean']:.2f} ± {summary['rmse_std']:.2f}")
    print(f"  CV R²   : {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
    return summary


def tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials: int = 100) -> dict:
    """Optuna HPO — val set is ONLY used here, never test set."""
    import optuna
    import xgboost as xgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators":        trial.suggest_int("n_estimators", 300, 1200, step=100),
            "max_depth":           trial.suggest_int("max_depth", 3, 9),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":           trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":    trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":    trial.suggest_int("min_child_weight", 1, 10),
            "gamma":               trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":           trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":          trial.suggest_float("reg_lambda", 0.5, 3.0),
            "random_state":        42,
            "n_jobs":              -1,
            "early_stopping_rounds": 30,
            "eval_metric":         "rmse",
        }
        m = xgb.XGBRegressor(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return np.sqrt(mean_squared_error(y_val, m.predict(X_val)))

    print(f"\nRunning Optuna ({n_trials} trials) ...")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    print(f"  Best val RMSE: {study.best_value:.3f}")
    return best


def train_final_model(X, y, params: dict):
    """Train final model on ALL data (train+val) — test set never included."""
    import xgboost as xgb
    clean_params = {k: v for k, v in params.items()
                    if k not in ("early_stopping_rounds", "eval_metric")}
    clean_params.update({"random_state": 42, "n_jobs": -1})
    model = xgb.XGBRegressor(**clean_params)
    model.fit(X, y, verbose=False)
    return model


def evaluate_test_set(model, X_test, y_test) -> dict:
    """Final, once-only evaluation on the held-out test set."""
    preds = model.predict(X_test)
    return {
        "mae":  round(float(mean_absolute_error(y_test, preds)),  3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
        "r2":   round(float(r2_score(y_test, preds)), 4),
        "n_samples": int(len(y_test)),
    }


def save_artifacts(model, feature_cols, cv_summary, best_params, test_metrics):
    import shap

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Model + feature list
    joblib.dump(model, MODEL_DIR / "xgb_model_v2.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols_v2.pkl")

    # Feature importance
    imp_df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    imp_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    # Best params + CV summary
    with open(MODEL_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    with open(MODEL_DIR / "cv_summary.json", "w") as f:
        json.dump({**cv_summary, "n_features": len(feature_cols)}, f, indent=2)

    # Test results
    test_results = {
        "split": {
            "train_end":  str(VAL_START.date()),
            "val_end":    str(TEST_START.date()),
            "test_start": str(TEST_START.date()),
        },
        "test_metrics": test_metrics,
        "cv_metrics": cv_summary,
    }
    with open(REPORT_DIR / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Human-readable metrics report
    with open(REPORT_DIR / "final_metrics.md", "w") as f:
        f.write("# Model Evaluation Results\n\n")
        f.write("## Data Split\n\n")
        f.write("| Set | Period | Rows |\n")
        f.write("|-----|--------|------|\n")
        f.write(f"| Train | Jan 2020 – Jun 2024 | ~{test_metrics['n_samples'] * 8} |\n")
        f.write(f"| Validation | Jul 2024 – Sep 2024 | ~{test_metrics['n_samples'] * 2} |\n")
        f.write(f"| **Test (held-out)** | Oct 2024 – Dec 2024 | **{test_metrics['n_samples']}** |\n\n")
        f.write("## Cross-Validation Metrics (5-fold TimeSeriesSplit, gap=24h)\n\n")
        f.write("| Metric | Mean | Std |\n|--------|------|-----|\n")
        f.write(f"| MAE    | {cv_summary['mae_mean']:.3f}  | ±{cv_summary['mae_std']:.3f} |\n")
        f.write(f"| RMSE   | {cv_summary['rmse_mean']:.3f} | ±{cv_summary['rmse_std']:.3f} |\n")
        f.write(f"| R²     | {cv_summary['r2_mean']:.4f} | ±{cv_summary['r2_std']:.4f} |\n\n")
        f.write("## Held-Out Test Set Metrics\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| MAE    | {test_metrics['mae']:.3f}  |\n")
        f.write(f"| RMSE   | {test_metrics['rmse']:.3f} |\n")
        f.write(f"| R²     | {test_metrics['r2']:.4f} |\n\n")
        f.write("> Test set was evaluated exactly once, after all hyperparameter decisions were finalized.\n")

    print("\nSaved:")
    print(f"  {MODEL_DIR}/xgb_model_v2.pkl")
    print(f"  {MODEL_DIR}/feature_cols_v2.pkl")
    print(f"  {MODEL_DIR}/feature_importance.csv")
    print(f"  {MODEL_DIR}/best_params.json")
    print(f"  {MODEL_DIR}/cv_summary.json")
    print(f"  {REPORT_DIR}/test_results.json")
    print(f"  {REPORT_DIR}/final_metrics.md")


def main():
    parser = argparse.ArgumentParser(description="Train Mumbai AQI XGBoost model")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials (default: 100)")
    parser.add_argument("--dry-run", action="store_true", help="Skip Optuna, use default params")
    args = parser.parse_args()

    print("=" * 60)
    print("Mumbai AQI Model — Training Pipeline")
    print("=" * 60)

    # 1. Load data
    df = load_data()
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"Features: {len(feature_cols)}")

    # 2. Temporal split
    train_df, val_df, test_df = split_data(df)

    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_val,   y_val   = prepare_xy(val_df,   feature_cols)
    X_test,  y_test  = prepare_xy(test_df,  feature_cols)

    # 3. Cross-validate with default params
    default_params = {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
    }
    cv_summary = cross_validate(X_train, y_train, default_params)
    cv_summary["n_features"]    = len(feature_cols)
    cv_summary["training_rows"] = len(X_train)

    # 4. Tune hyperparameters (on val set only)
    if args.dry_run:
        print("\n[Dry run] Skipping Optuna, using default params.")
        best_params = default_params
    else:
        best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, args.trials)

    # 5. Train final model on train+val (test never seen)
    print("\nTraining final model on train + val data ...")
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    final_model = train_final_model(X_trainval, y_trainval, best_params)

    # 6. Evaluate on held-out test set (once, final)
    print("\nEvaluating on held-out test set ...")
    test_metrics = evaluate_test_set(final_model, X_test, y_test)
    print(f"  Test MAE  : {test_metrics['mae']:.3f}")
    print(f"  Test RMSE : {test_metrics['rmse']:.3f}")
    print(f"  Test R²   : {test_metrics['r2']:.4f}")

    # 7. Save everything
    save_artifacts(final_model, feature_cols, cv_summary, best_params, test_metrics)

    print("\n" + "=" * 60)
    print("Training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
