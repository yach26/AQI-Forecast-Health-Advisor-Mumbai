import requests
import pandas as pd

url = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=19.0760"
    "&longitude=72.8777"
    "&start_date=2025-06-13"
    "&end_date=2026-06-13"
    "&hourly=temperature_2m,"
    "relative_humidity_2m,"
    "precipitation,"
    "wind_speed_10m,"
    "wind_direction_10m,"
    "surface_pressure"
    "&timezone=Asia/Kolkata"
)

response = requests.get(url)

print(response.status_code)

data = response.json()

weather_df = pd.DataFrame(data["hourly"])

print(weather_df.head())
print(weather_df.shape)

weather_df.to_csv("mumbai_weather_1year.csv", index=False)

print("Weather CSV saved!")