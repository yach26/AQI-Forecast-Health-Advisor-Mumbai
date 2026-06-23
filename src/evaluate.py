"""
src/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Standalone evaluation script. Loads the saved model and evaluates it against
a specified date range from the dataset, producing detailed metrics.

Usage
─────
    python src/evaluate.py                         # evaluates test set (Oct–Dec 2024)
    python src/evaluate.py --start 2024-07-01      # custom start date
    python src/evaluate.py --start 2024-07-01 --end 2024-09-30

Outputs to console + saves to reports/evaluation_report.json
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

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data" / "mumbai_aqi_5yr_expanded.csv"
MODEL_DIR  = ROOT / "notebooks" / "model"
REPORT_DIR = ROOT / "reports"

EXCLUDE_COLS = [
    "time", "us_aqi", "us_aqi_pm2_5", "us_aqi_pm10",
    "us_aqi_nitrogen_dioxide", "us_aqi_ozone", "season",
]
TARGET = "us_aqi"


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Mumbai AQI model")
    parser.add_argument("--start", default="2024-10-01", help="Eval start date (YYYY-MM-DD)")
    parser.add_argument("--end",   default="2024-12-31", help="Eval end date (YYYY-MM-DD)")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = MODEL_DIR / "xgb_model_v2.pkl"
    feat_path  = MODEL_DIR / "feature_cols_v2.pkl"

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run: python src/train.py")
        sys.exit(1)

    model        = joblib.load(model_path)
    feature_cols = joblib.load(feat_path)
    print(f"Model loaded: {model_path.name}")
    print(f"Features    : {len(feature_cols)}")

    # ── Load data ─────────────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Run: python src/build_dataset.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # ── Filter to evaluation window ───────────────────────────────────────────
    mask = (df["time"] >= args.start) & (df["time"] <= args.end)
    eval_df = df[mask].copy()

    if eval_df.empty:
        print(f"ERROR: No data found for {args.start} → {args.end}")
        sys.exit(1)

    print(f"\nEvaluation period: {eval_df['time'].min().date()} → {eval_df['time'].max().date()}")
    print(f"Rows: {len(eval_df):,}")

    # ── Prepare features ──────────────────────────────────────────────────────
    available_features = [c for c in feature_cols if c in eval_df.columns]
    missing_features   = [c for c in feature_cols if c not in eval_df.columns]

    if missing_features:
        print(f"\nWarning: {len(missing_features)} features missing, filling with 0:")
        for f in missing_features:
            eval_df[f] = 0.0

    eval_clean = eval_df[feature_cols + [TARGET]].dropna()
    X_eval = eval_clean[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y_eval = eval_clean[TARGET].values

    # ── Predict ───────────────────────────────────────────────────────────────
    y_pred = model.predict(X_eval)

    # ── Metrics ───────────────────────────────────────────────────────────────
    mae  = mean_absolute_error(y_eval, y_pred)
    rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
    r2   = r2_score(y_eval, y_pred)

    # Per-AQI-category breakdown
    categories = pd.cut(
        y_eval,
        bins=[0, 50, 100, 150, 200, 300, 500],
        labels=["Good", "Moderate", "Sensitive", "Unhealthy", "Very Unhealthy", "Hazardous"],
    )
    breakdown = {}
    for cat in categories.unique():
        if pd.isna(cat):
            continue
        mask_cat = categories == cat
        if mask_cat.sum() < 5:
            continue
        breakdown[str(cat)] = {
            "n":    int(mask_cat.sum()),
            "mae":  round(float(mean_absolute_error(y_eval[mask_cat], y_pred[mask_cat])), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_eval[mask_cat], y_pred[mask_cat]))), 2),
        }

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print("Evaluation Results")
    print("=" * 45)
    print(f"  MAE   : {mae:.3f}")
    print(f"  RMSE  : {rmse:.3f}")
    print(f"  R²    : {r2:.4f}")
    print(f"  Samples: {len(y_eval):,}")
    print("\nBreakdown by AQI Category:")
    for cat, stats in breakdown.items():
        print(f"  {cat:25s} n={stats['n']:5,}  MAE={stats['mae']:.2f}  RMSE={stats['rmse']:.2f}")

    # ── Save report ───────────────────────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "period": {"start": args.start, "end": args.end},
        "overall": {"mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 4), "n": len(y_eval)},
        "by_category": breakdown,
    }
    out = REPORT_DIR / "evaluation_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()
