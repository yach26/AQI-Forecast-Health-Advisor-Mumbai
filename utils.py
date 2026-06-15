"""
Utility module for Mumbai AQI Forecast & AI Health Advisor.

Handles data fetching from Open-Meteo APIs, feature engineering,
model loading, and AQI prediction pipeline.
"""

import requests
import pandas as pd
import numpy as np
import json
import joblib
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the app works from any CWD
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "aqi_model_24h.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

# ---------------------------------------------------------------------------
# Mumbai coordinates
# ---------------------------------------------------------------------------
MUMBAI_LAT = 19.0760
MUMBAI_LON = 72.8777

# ---------------------------------------------------------------------------
# Model uncertainty (RMSE from training evaluation)
# ---------------------------------------------------------------------------
MODEL_RMSE = 15


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained XGBoost model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Ensure 'model/aqi_model_24h.pkl' exists."
        )
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_feature_columns():
    """Load the list of feature column names used during training."""
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Feature columns file not found: {FEATURE_COLUMNS_PATH}. "
            "Ensure 'model/feature_columns.json' exists."
        )
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_importance():
    """Load feature importance CSV for visualization."""
    if not FEATURE_IMPORTANCE_PATH.exists():
        return None
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


# ---------------------------------------------------------------------------
# AQI category classification
# ---------------------------------------------------------------------------
def get_category(aqi):
    """Classify an AQI value into a human-readable category string."""
    aqi = float(aqi)

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# ---------------------------------------------------------------------------
# API data fetching — 7-day history window for robust lag/rolling features
# ---------------------------------------------------------------------------
def fetch_live_data():
    """
    Fetch live air-quality and weather data from Open-Meteo APIs.

    Uses a 7-day history window (``past_days=7``) so that lag and rolling
    features (up to 48-hour lags and 24-hour rolling windows) have
    sufficient history even after NaN rows are dropped.
    """
    air_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={MUMBAI_LAT}"
        f"&longitude={MUMBAI_LON}"
        f"&hourly=us_aqi,pm2_5,pm10,"
        f"nitrogen_dioxide,ozone,"
        f"sulphur_dioxide,carbon_monoxide"
        f"&past_days=7"
    )

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={MUMBAI_LAT}"
        f"&longitude={MUMBAI_LON}"
        f"&hourly=temperature_2m,"
        f"relative_humidity_2m,"
        f"precipitation,"
        f"wind_speed_10m,"
        f"wind_direction_10m,"
        f"surface_pressure"
        f"&past_days=7"
    )

    try:
        air_response = requests.get(air_url, timeout=30)
        air_response.raise_for_status()

        weather_response = requests.get(weather_url, timeout=30)
        weather_response.raise_for_status()

    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Failed to fetch data from Open-Meteo APIs: {e}"
        ) from e

    air_df = pd.DataFrame(air_response.json()["hourly"])
    weather_df = pd.DataFrame(weather_response.json()["hourly"])

    df = pd.merge(air_df, weather_df, on="time")

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df):
    """
    Build the full feature set expected by the trained XGBoost model.

    Steps:
        1. Parse timestamps and extract temporal features.
        2. Add cyclical sin/cos encodings for hour and month.
        3. Compute AQI lag features (1, 3, 6, 12, 24, 48 hours).
        4. Compute 24-hour rolling mean/std for AQI and PM2.5.
        5. Forward-fill then drop remaining NaN rows.

    Using ``ffill()`` before ``dropna()`` reduces data loss from
    intermittent API gaps while still removing rows where lag/rolling
    features cannot be computed (e.g. the first 48 rows).
    """
    df = df.copy()

    df["time"] = pd.to_datetime(df["time"])

    # Temporal Features
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["week_of_year"] = (
        df["time"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features
    df["aqi_lag_1"] = df["us_aqi"].shift(1)
    df["aqi_lag_3"] = df["us_aqi"].shift(3)
    df["aqi_lag_6"] = df["us_aqi"].shift(6)
    df["aqi_lag_12"] = df["us_aqi"].shift(12)
    df["aqi_lag_24"] = df["us_aqi"].shift(24)
    df["aqi_lag_48"] = df["us_aqi"].shift(48)

    # Rolling statistics
    df["aqi_roll_mean_24"] = (
        df["us_aqi"]
        .rolling(window=24, min_periods=12)
        .mean()
    )

    df["aqi_roll_std_24"] = (
        df["us_aqi"]
        .rolling(window=24, min_periods=12)
        .std()
    )

    df["pm25_roll_mean_24"] = (
        df["pm2_5"]
        .rolling(window=24, min_periods=12)
        .mean()
    )

    # Fill intermittent gaps, then drop rows that still have NaNs
    # (the initial rows where lag/rolling features are structurally absent)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    return df


# ---------------------------------------------------------------------------
# Main prediction entry point
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def predict_aqi():
    """
    End-to-end prediction pipeline.

    Returns a dict with:
        - current_aqi: int
        - forecast_aqi: int
        - category: str
        - timestamp: pd.Timestamp
        - forecast_timestamp: pd.Timestamp
        - confidence_lower: int
        - confidence_upper: int
        - history: pd.DataFrame (last 48 rows of raw data)
    """
    model = load_model()
    feature_columns = load_feature_columns()

    raw_df = fetch_live_data()

    recent_history = raw_df.copy().tail(48)

    df = engineer_features(raw_df)

    if df.empty:
        raise ValueError(
            "Feature engineering produced an empty DataFrame. "
            "The Open-Meteo APIs may be returning incomplete data."
        )

    latest = df.iloc[-1:]

    # Validate that all expected features are present
    missing_cols = set(feature_columns) - set(latest.columns)
    if missing_cols:
        raise ValueError(
            f"Missing features after engineering: {missing_cols}. "
            "The model cannot produce a prediction."
        )

    X_live = latest[feature_columns]

    prediction = model.predict(X_live)[0]
    prediction = round(float(prediction))

    current_aqi = int(latest["us_aqi"].iloc[0])
    timestamp = latest["time"].iloc[0]
    forecast_timestamp = timestamp + pd.Timedelta(hours=24)
    category = get_category(prediction)

    return {
        "current_aqi": current_aqi,
        "forecast_aqi": prediction,
        "category": category,
        "timestamp": timestamp,
        "forecast_timestamp": forecast_timestamp,
        "confidence_lower": max(0, prediction - MODEL_RMSE),
        "confidence_upper": prediction + MODEL_RMSE,
        "history": recent_history,
    }