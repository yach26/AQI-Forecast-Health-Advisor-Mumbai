import requests
import pandas as pd
import numpy as np
import json
import joblib

model = joblib.load("model/aqi_model_24h.pkl")

with open("model/feature_columns.json", "r") as f:
    feature_columns = json.load(f)


def get_category(aqi):

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


def fetch_live_data():

    latitude = 19.0760
    longitude = 72.8777

    air_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        f"&hourly=us_aqi,pm2_5,pm10,"
        f"nitrogen_dioxide,ozone,"
        f"sulphur_dioxide,carbon_monoxide"
        f"&past_days=3"
    )

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        f"&hourly=temperature_2m,"
        f"relative_humidity_2m,"
        f"precipitation,"
        f"wind_speed_10m,"
        f"wind_direction_10m,"
        f"surface_pressure"
        f"&past_days=3"
    )

    try:
        air_response = requests.get(air_url, timeout=30)
        air_response.raise_for_status()

        weather_response = requests.get(weather_url, timeout=30)
        weather_response.raise_for_status()

    except requests.exceptions.RequestException as e:
        raise Exception(f"API Error: {e}")

    air_df = pd.DataFrame(air_response.json()["hourly"])
    weather_df = pd.DataFrame(weather_response.json()["hourly"])

    df = pd.merge(air_df, weather_df, on="time")

    return df


def engineer_features(df):

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

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["aqi_lag_1"] = df["us_aqi"].shift(1)
    df["aqi_lag_3"] = df["us_aqi"].shift(3)
    df["aqi_lag_6"] = df["us_aqi"].shift(6)
    df["aqi_lag_12"] = df["us_aqi"].shift(12)
    df["aqi_lag_24"] = df["us_aqi"].shift(24)
    df["aqi_lag_48"] = df["us_aqi"].shift(48)

    df["aqi_roll_mean_24"] = (
        df["us_aqi"]
        .rolling(window=24)
        .mean()
    )

    df["aqi_roll_std_24"] = (
        df["us_aqi"]
        .rolling(window=24)
        .std()
    )

    df["pm25_roll_mean_24"] = (
        df["pm2_5"]
        .rolling(window=24)
        .mean()
    )

    df.dropna(inplace=True)

    return df


def predict_aqi():

    raw_df = fetch_live_data()

    recent_history = raw_df.copy().tail(48)

    df = engineer_features(raw_df)

    latest = df.iloc[-1:]

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
        "history": recent_history
    }