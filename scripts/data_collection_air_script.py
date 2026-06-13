import requests
import pandas as pd
import os

print("Script started...")

url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    "?latitude=19.0760"
    "&longitude=72.8777"
    "&hourly=us_aqi,pm2_5,pm10,nitrogen_dioxide,"
    "ozone,sulphur_dioxide,carbon_monoxide"
    "&start_date=2025-06-13"
    "&end_date=2026-06-13"
    "&timezone=Asia/Kolkata"
)

print("Fetching data...")

response = requests.get(url)

print("Status Code:", response.status_code)

data = response.json()

print("Response keys:", data.keys())

df = pd.DataFrame(data["hourly"])

print("Shape:", df.shape)
print(df.head())

output_file = "mumbai_air_quality_1year.csv"

df.to_csv(output_file, index=False)

print("CSV saved at:", os.path.abspath(output_file))