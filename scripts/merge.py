import pandas as pd
aqi = pd.read_csv("mumbai_air_quality_1year.csv")
weather = pd.read_csv("mumbai_weather_1year.csv")

df = pd.merge(aqi, weather, on="time", how="inner")

print(df.shape)

df.to_csv("mumbai_final_dataset.csv", index=False)