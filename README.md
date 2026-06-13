# AQI Forecast & Health Advisor — Mumbai

This repository contains data, scripts, and a notebook used to explore and model air quality (AQI) and related weather data for Mumbai.

Repository layout
- data/: CSV datasets used for analysis and modeling
- model/: Jupyter notebook(s) with EDA and modeling (`aqi-weather-data-mumbai-eda.ipynb`)
- scripts/: Data collection and preprocessing scripts (`data_collection_air_script.py`, `data_collection_weather_script.py`, `merge.py`)

Quick start
1. Create and activate a Python virtual environment:

   python -m venv .venv
   .\.venv\Scripts\activate

2. Install dependencies:

   pip install -r requirements.txt

3. Explore the notebook in `model/` or run the scripts in `scripts/` to reproduce data processing.

Notes
- Keep raw data in `data/`. Small sample CSVs are already included.
- Update `requirements.txt` when adding new libraries.

License
This project does not include a license file. Add one if you plan to publish or share the code.
