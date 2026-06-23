"""
Utility module for Mumbai AQI Forecast & AI Health Advisor.

Handles data fetching from Open-Meteo APIs, feature engineering,
model loading, and AQI prediction pipeline.
"""

import requests
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the app works from any CWD
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "notebooks" / "model"

MODEL_PATH = MODEL_DIR / "xgb_model_v2.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_cols_v2.pkl"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

# ---------------------------------------------------------------------------
# Mumbai coordinates
# ---------------------------------------------------------------------------
MUMBAI_LAT = 19.0760
MUMBAI_LON = 72.8777

# ---------------------------------------------------------------------------
# Model uncertainty (RMSE from training evaluation)
# ---------------------------------------------------------------------------
MODEL_RMSE = 5.11  # RMSE updated from v2 training cross-validation


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained XGBoost model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Ensure 'notebooks/model/xgb_model_v2.pkl' exists."
        )
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_feature_columns():
    """Load the list of feature column names used during training."""
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Feature columns file not found: {FEATURE_COLUMNS_PATH}. "
            "Ensure 'notebooks/model/feature_cols_v2.pkl' exists."
        )
    return joblib.load(FEATURE_COLUMNS_PATH)


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
    features (up to 72-hour lags and 48-hour rolling windows) have
    sufficient history even after NaN rows are dropped.
    """
    air_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={MUMBAI_LAT}"
        f"&longitude={MUMBAI_LON}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,ammonia,us_aqi,us_aqi_pm2_5,us_aqi_pm10,us_aqi_nitrogen_dioxide,us_aqi_ozone,dust"
        f"&past_days=7"
    )

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={MUMBAI_LAT}"
        f"&longitude={MUMBAI_LON}"
        f"&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,visibility,uv_index,boundary_layer_height,cloud_cover,et0_fao_evapotranspiration,shortwave_radiation"
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
    Build the full 80-feature set expected by the trained XGBoost v2 model.
    """
    df = df.copy()

    df["time"] = pd.to_datetime(df["time"])

    # 1. Temporal features
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["week_of_year"] = df["time"].dt.isocalendar().week.astype(int)
    df["year"] = df["time"].dt.year
    df["day_of_year"] = df["time"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # 2. Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # 3. Lag features
    for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
        df[f"aqi_lag_{lag}"] = df["us_aqi"].shift(lag)

    for lag in [1, 3, 6, 24]:
        df[f"pm25_lag_{lag}"] = df["pm2_5"].shift(lag)

    for lag in [1, 3, 6, 12, 24]:
        df[f"precip_lag_{lag}"] = df["precipitation"].shift(lag)

    for lag in [1, 6]:
        df[f"wind_lag_{lag}"] = df["wind_speed_10m"].shift(lag)

    # 4. Rolling statistics
    df["aqi_roll_mean_3"] = df["us_aqi"].rolling(3, min_periods=1).mean()
    df["aqi_roll_mean_6"] = df["us_aqi"].rolling(6, min_periods=1).mean()
    df["aqi_roll_mean_12"] = df["us_aqi"].rolling(12, min_periods=1).mean()
    df["aqi_roll_mean_24"] = df["us_aqi"].rolling(24, min_periods=1).mean()
    df["aqi_roll_mean_48"] = df["us_aqi"].rolling(48, min_periods=1).mean()
    df["aqi_roll_std_24"] = df["us_aqi"].rolling(24, min_periods=1).std()
    df["aqi_roll_max_24"] = df["us_aqi"].rolling(24, min_periods=1).max()
    df["aqi_roll_min_24"] = df["us_aqi"].rolling(24, min_periods=1).min()

    df["pm25_roll_mean_24"] = df["pm2_5"].rolling(24, min_periods=1).mean()
    df["pm25_roll_max_24"] = df["pm2_5"].rolling(24, min_periods=1).max()

    df["precip_sum_6h"] = df["precipitation"].rolling(6, min_periods=1).sum()
    df["precip_sum_24h"] = df["precipitation"].rolling(24, min_periods=1).sum()

    # 5. Mumbai Season flag
    def get_mumbai_season(month):
        if month in [6, 7, 8, 9]:
            return "monsoon"
        elif month in [10, 11]:
            return "post_monsoon"
        elif month in [12, 1, 2]:
            return "winter"
        else:
            return "summer"

    df["season"] = df["month"].apply(get_mumbai_season)

    season_dummies = pd.get_dummies(df["season"], prefix="season", drop_first=False, dtype=int)
    for col in ["season_post_monsoon", "season_summer", "season_winter"]:
        if col not in season_dummies.columns:
            season_dummies[col] = 0
    df = pd.concat([df, season_dummies], axis=1)
    df.drop(columns=["season_monsoon"], errors="ignore", inplace=True)

    # 6. Festival / Pollution-event flags
    DIWALI_DATES = ["2020-11-14", "2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01"]
    HOLI_DATES = ["2020-03-10", "2021-03-29", "2022-03-18", "2023-03-08", "2024-03-25"]
    GANESH_DATES = ["2020-09-01", "2021-09-19", "2022-09-09", "2023-09-28", "2024-09-17"]

    festival_dates = set(DIWALI_DATES + HOLI_DATES + GANESH_DATES + [
        "2025-10-20", "2026-11-08", "2025-03-14", "2026-03-03", "2025-08-27", "2026-09-15"
    ])
    festival_date_dt = set(pd.to_datetime(list(festival_dates)).date)
    festival_plus1 = set([(d + pd.Timedelta("1D")).date() for d in pd.to_datetime(list(festival_dates))])
    all_festival_days = festival_date_dt | festival_plus1

    df["is_festival"] = df["time"].dt.date.isin(all_festival_days).astype(int)

    # 7. Interaction / derived features
    if "boundary_layer_height" in df.columns:
        df["ventilation_idx"] = df["boundary_layer_height"] * df["wind_speed_10m"] / 1000
        df["inversion_proxy"] = (df["boundary_layer_height"] < 500).astype(int)
    else:
        df["ventilation_idx"] = 0
        df["inversion_proxy"] = 0

    df["humidity_pm25"] = df["relative_humidity_2m"] * df["pm2_5"] / 100

    # Wind direction dummies
    df["wind_dir_cat"] = pd.cut(
        df["wind_direction_10m"],
        bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
        labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        right=False,
    ).astype(str)

    wind_dummies = pd.get_dummies(df["wind_dir_cat"], prefix="wdir", drop_first=False, dtype=int)
    for col in ["wdir_N", "wdir_NE", "wdir_NW", "wdir_S", "wdir_SE", "wdir_SW", "wdir_W"]:
        if col not in wind_dummies.columns:
            wind_dummies[col] = 0
    df = pd.concat([df, wind_dummies], axis=1)
    df.drop(columns=["wind_dir_cat"], errors="ignore", inplace=True)

    # Fill intermittent gaps, then drop remaining NaNs (lag warmups)
    non_target_cols = [c for c in df.columns if c not in ["us_aqi", "time"]]
    df[non_target_cols] = df[non_target_cols].ffill(limit=3)
    df[non_target_cols] = df[non_target_cols].fillna(0)
    df.dropna(subset=non_target_cols, inplace=True)

    return df


# ---------------------------------------------------------------------------
# Main prediction entry point
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def predict_aqi():
    """
    End-to-end prediction pipeline.
    """
    model = load_model()
    feature_columns = load_feature_columns()

    raw_df = fetch_live_data()

    df = engineer_features(raw_df)

    if df.empty:
        raise ValueError(
            "Feature engineering produced an empty DataFrame. "
            "The Open-Meteo APIs may be returning incomplete data."
        )

    # Filter to only rows at or before now, then take the last one:
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    df_past = df[df["time"] <= now]

    if df_past.empty:
        raise ValueError("No past data rows found after filtering to current time.")

    latest = df_past.iloc[-1:]

    # Dynamically align features to match training feature set exactly
    latest_aligned = pd.DataFrame(index=[0])
    for col in feature_columns:
        if col in latest.columns:
            latest_aligned[col] = latest[col].values[0]
        else:
            latest_aligned[col] = 0.0

    # Ensure all features are cast to float/numeric to satisfy XGBoost validation
    X_live = latest_aligned[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    prediction = model.predict(X_live)[0]
    prediction = round(float(prediction))

    current_aqi = int(latest["us_aqi"].iloc[0])
    timestamp = latest["time"].iloc[0]
    forecast_timestamp = timestamp + pd.Timedelta(hours=24)
    category = get_category(prediction)

    # Filter recent history to past 48 hours relative to the current timestamp:
    raw_df_dt = raw_df.copy()
    raw_df_dt["time"] = pd.to_datetime(raw_df_dt["time"])
    recent_history = raw_df_dt[raw_df_dt["time"] <= timestamp].tail(48)

    return {
        "current_aqi": current_aqi,
        "forecast_aqi": prediction,
        "category": category,
        "timestamp": timestamp,
        "forecast_timestamp": forecast_timestamp,
        "confidence_lower": max(0, prediction - round(MODEL_RMSE)),
        "confidence_upper": prediction + round(MODEL_RMSE),
        "history": recent_history,
    }