import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import predict_aqi
from llm_utils import generate_health_advice

# 1. Page Configuration
st.set_page_config(
    page_title="Mumbai AQI Forecast & Health Advisor",
    layout="wide",
    page_icon="🌫️"
)

# Title
st.title("Mumbai AQI Forecast & Health Advisor")
st.caption("Live 24-Hour AQI Forecast powered by XGBoost and Open-Meteo APIs")

# Refresh button
if st.button("Refresh Live Data"):
    st.rerun()

# Loading experience
with st.spinner("Fetching latest Mumbai air quality data..."):
    data = predict_aqi()

st.success("Live forecast updated successfully.")

current_aqi = data["current_aqi"]
forecast_aqi = data["forecast_aqi"]
category = data["category"]
timestamp = data["timestamp"]
forecast_timestamp = data["forecast_timestamp"]
history = data["history"]

delta = forecast_aqi - current_aqi

# 1. Forecast Confidence Section
lower = max(0, forecast_aqi - 15)
upper = forecast_aqi + 15

# Metrics block covering: 2. Current AQI, 3. Forecast AQI, 4. AQI Category, 5. Forecast Confidence
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Current AQI",
        value=current_aqi
    )

with col2:
    st.metric(
        label="Forecast AQI",
        value=forecast_aqi,
        delta=delta
    )

with col3:
    st.write("**AQI Category**")
    if category == "Good":
        st.success("Good")
    elif category == "Moderate":
        st.info("Moderate")
    elif category == "Unhealthy for Sensitive Groups":
        st.warning("Unhealthy for Sensitive Groups")
    else:
        st.error(category)

with col4:
    st.metric(
        label="Confidence Range",
        value=f"{lower} to {upper}"
    )

# 6. AQI Gauge
st.write("")
fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = forecast_aqi,
    domain = {'x': [0, 1], 'y': [0, 1]},
    gauge = {
        'axis': {'range': [0, 400], 'tickwidth': 1, 'tickcolor': "gray"},
        'bar': {'color': "#34495e"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 50], 'color': '#2ecc71'},
            {'range': [50, 100], 'color': '#f1c40f'},
            {'range': [100, 150], 'color': '#e67e22'},
            {'range': [150, 200], 'color': '#e74c3c'},
            {'range': [200, 300], 'color': '#9b59b6'},
            {'range': [300, 400], 'color': '#78281f'}
        ],
    }
))
fig_gauge.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=280,
    margin=dict(l=30, r=30, t=30, b=30),
)
st.plotly_chart(fig_gauge, use_container_width=True)

# 7. AQI Trend (Last 48 Hours)
st.subheader("Mumbai AQI Trend (Last 48 Hours)")
history_copy = history.copy()
history_copy["time"] = history_copy["time"].astype(str)
fig_trend = px.line(
    history_copy,
    x="time",
    y="us_aqi",
    title="Mumbai AQI Trend (Last 48 Hours)"
)
fig_trend.update_layout(
    xaxis_title="Timestamp",
    yaxis_title="AQI"
)
st.plotly_chart(fig_trend, use_container_width=True)

# 8. Current Pollutant Levels
st.subheader("Current Pollutant Levels")
latest_row = history.iloc[-1]
pm25_val = latest_row["pm2_5"]
pm10_val = latest_row["pm10"]
no2_val = latest_row["nitrogen_dioxide"]

col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    st.metric(label="PM2.5", value=f"{pm25_val:.1f} µg/m³")
with col_p2:
    st.metric(label="PM10", value=f"{pm10_val:.1f} µg/m³")
with col_p3:
    st.metric(label="NO₂", value=f"{no2_val:.1f} µg/m³")

# 9. AI Health Advisor
st.subheader("AI Health Advisor")
with st.container(border=True):
    with st.spinner("Generating personalized recommendations..."):
        advice = generate_health_advice(
            forecast_aqi,
            category
        )
    st.markdown(advice)

# 10. Top Drivers Behind AQI Predictions
st.subheader("Top Drivers Behind AQI Predictions")
df_importance = pd.read_csv("model/feature_importance.csv")
df_importance_sorted = df_importance.sort_values(by="Importance", ascending=True)
st.bar_chart(df_importance_sorted, x="Importance", y="Feature", horizontal=True)

# 11. About This Forecasting Model
with st.expander("About This Forecasting Model"):
    st.markdown(
        """
**Model:** XGBoost Regressor

**Forecast Horizon:** 24 Hours Ahead

**Training Data:** 1 Year of Mumbai AQI + Weather Data

**Evaluation Metrics:**
* RMSE ≈ 15
* R² ≈ 0.62

**Features Used:**
* Historical AQI lags
* Rolling statistics
* Pollutant concentrations
* Weather variables
* Temporal features
* Cyclical encodings

**Data Sources:**
* Open-Meteo Air Quality API
* Open-Meteo Weather API
"""
    )

st.write("---")

# 12. Last Updated & 13. Forecast Valid For
st.write(
    f"**Last Updated:** {timestamp.strftime('%d %b %Y, %I:%M %p')}"
)
st.write(
    f"**Forecast Valid For:** {forecast_timestamp.strftime('%d %b %Y, %I:%M %p')}"
)

# Sidebar
st.sidebar.title("Project Overview")
st.sidebar.write(
    "This application predicts Mumbai's AQI 24 hours ahead using a machine learning model trained on historical AQI and weather data."
)

st.sidebar.subheader("Technology Stack")
st.sidebar.markdown(
    """
- Streamlit
- XGBoost
- Open-Meteo APIs
- Pandas
- Plotly
"""
)

st.sidebar.subheader("Future Roadmap")
st.sidebar.markdown(
    """
- Multi-city forecasting
- Groq-powered personalized health advisor
- 48-hour and 72-hour forecasts
- Push notifications
"""
)