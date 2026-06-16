import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import predict_aqi, load_feature_importance
from llm_utils import generate_health_advice


st.set_page_config(
    page_title="Mumbai AQI Forecast & Health Advisor",
    layout="wide",
    page_icon="🌫️",
)


st.title("Mumbai AQI Forecast & Health Advisor")
st.caption("Live 24-Hour AQI Forecast powered by XGBoost and Open-Meteo APIs")

if st.button("Refresh Live Data"):
    st.cache_data.clear()
    st.rerun()

try:
    with st.spinner("Fetching latest Mumbai air quality data..."):
        data = predict_aqi()
    st.success("Live forecast updated successfully.")
except Exception as e:
    st.error(f"Could not load forecast data: {e}")
    st.info("Please check your internet connection and try refreshing.")
    st.stop()

current_aqi = data["current_aqi"]
forecast_aqi = data["forecast_aqi"]
category = data["category"]
timestamp = data["timestamp"]
forecast_timestamp = data["forecast_timestamp"]
history = data["history"]
confidence_lower = data["confidence_lower"]
confidence_upper = data["confidence_upper"]

delta = forecast_aqi - current_aqi

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Current AQI", value=current_aqi)

with col2:
    st.metric(label="Forecast AQI", value=forecast_aqi, delta=delta)

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
        value=f"{confidence_lower} – {confidence_upper}",
    )


st.write("")
fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=forecast_aqi,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 400], "tickwidth": 1, "tickcolor": "gray"},
            "bar": {"color": "#34495e"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 50], "color": "#2ecc71"},
                {"range": [50, 100], "color": "#f1c40f"},
                {"range": [100, 150], "color": "#e67e22"},
                {"range": [150, 200], "color": "#e74c3c"},
                {"range": [200, 300], "color": "#9b59b6"},
                {"range": [300, 400], "color": "#78281f"},
            ],
        },
    )
)
fig_gauge.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=280,
    margin=dict(l=30, r=30, t=30, b=30),
)
st.plotly_chart(fig_gauge, width="stretch")


st.subheader("Mumbai AQI Trend (Last 48 Hours)")
history_copy = history.copy()
history_copy["time"] = pd.to_datetime(history_copy["time"])
fig_trend = px.line(
    history_copy,
    x="time",
    y="us_aqi",
    labels={"time": "Timestamp", "us_aqi": "AQI"},
)
fig_trend.update_layout(xaxis_title="Timestamp", yaxis_title="AQI")
st.plotly_chart(fig_trend, width="stretch")

st.subheader("Current Pollutant Levels")
latest_row = history.iloc[-1]

col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    st.metric(label="PM2.5", value=f"{latest_row['pm2_5']:.1f} µg/m³")
with col_p2:
    st.metric(label="PM10", value=f"{latest_row['pm10']:.1f} µg/m³")
with col_p3:
    st.metric(label="NO₂", value=f"{latest_row['nitrogen_dioxide']:.1f} µg/m³")


st.subheader("AI Health Advisor")
with st.container(border=True):
    with st.spinner("Generating personalized recommendations..."):
        advice = generate_health_advice(forecast_aqi, category)
    st.markdown(advice)

st.subheader("Top 10 Drivers Behind AQI Predictions")
df_importance = load_feature_importance()
if df_importance is not None and not df_importance.empty:
    df_sorted = df_importance.sort_values(by="Importance", ascending=True)
    fig_imp = px.bar(
        df_sorted,
        x="Importance",
        y="Feature",
        orientation="h",
        labels={"Importance": "Feature Importance Score", "Feature": ""},
    )
    fig_imp.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=350,
        yaxis=dict(tickfont=dict(size=12)),
    )
    st.plotly_chart(fig_imp, width="stretch")
else:
    st.info("Feature importance data is not available.")


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


st.write(f"**Last Updated:** {timestamp.strftime('%d %b %Y, %I:%M %p')}")
st.write(
    f"**Forecast Valid For:** {forecast_timestamp.strftime('%d %b %Y, %I:%M %p')}"
)

st.sidebar.title("Project Overview")
st.sidebar.write(
    "This application predicts Mumbai's AQI 24 hours ahead using a "
    "machine learning model trained on historical AQI and weather data."
)

st.sidebar.subheader("Technology Stack")
st.sidebar.markdown(
    """
- Streamlit
- XGBoost
- Groq (Llama-3.3)
- Open-Meteo APIs
- Pandas · Plotly
"""
)

st.sidebar.subheader("Future Roadmap")
st.sidebar.markdown(
    """
- Multi-city forecasting
- 48-hour and 72-hour forecasts
- Push notifications
- Mobile optimization
"""
)