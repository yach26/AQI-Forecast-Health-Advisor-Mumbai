import requests
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta

LAT = 19.0760
LON = 72.8777


def fetch_air_data():
    end = datetime.utcnow().date()
    start = end - timedelta(days=3)

    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={LAT}"
        f"&longitude={LON}"
        f"&start_date={start}"
        f"&end_date={end}"
        f"&hourly=us_aqi,pm2_5,pm10,nitrogen_dioxide,"
        f"ozone,sulphur_dioxide,carbon_monoxide"
    )

    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data["hourly"])

    return df


def fetch_weather_data():
    end = datetime.utcnow().date()
    start = end - timedelta(days=3)

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}"
        f"&longitude={LON}"
        f"&start_date={start}"
        f"&end_date={end}"
        f"&hourly=temperature_2m,relative_humidity_2m,"
        f"precipitation,wind_speed_10m,"
        f"wind_direction_10m,surface_pressure"
    )

    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data["hourly"])

    return df


def engineer_features(df):
    df["time"] = pd.to_datetime(df["time"])

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["week_of_year"] = df["time"].dt.isocalendar().week.astype(int)

    # Lag Features
    df["aqi_lag_1"] = df["us_aqi"].shift(1)
    df["aqi_lag_3"] = df["us_aqi"].shift(3)
    df["aqi_lag_6"] = df["us_aqi"].shift(6)
    df["aqi_lag_12"] = df["us_aqi"].shift(12)
    df["aqi_lag_24"] = df["us_aqi"].shift(24)
    df["aqi_lag_48"] = df["us_aqi"].shift(48)

    # Rolling Features
    df["aqi_roll_mean_24"] = (
        df["us_aqi"]
        .rolling(24)
        .mean()
    )

    df["aqi_roll_std_24"] = (
        df["us_aqi"]
        .rolling(24)
        .std()
    )

    df["pm25_roll_mean_24"] = (
        df["pm2_5"]
        .rolling(24)
        .mean()
    )

    # Cyclical Features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def get_category(aqi):
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

    return "Hazardous"


print("Fetching live data...")

air = fetch_air_data()
weather = fetch_weather_data()

df = air.merge(weather, on="time")

df = engineer_features(df)

df = df.dropna()

latest = df.iloc[-1:]

print("\nLatest Timestamp:")
print(latest["time"].values[0])

with open("model/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

X_live = latest[feature_columns]

model = joblib.load("model/aqi_model_24h.pkl")

prediction = model.predict(X_live)[0]

current_aqi = latest["us_aqi"].values[0]

print("\nCurrent AQI:", round(current_aqi))
print("Forecast AQI (24h ahead):", round(prediction))

print(
    "Forecast Category:",
    get_category(prediction)
)