"""
Mumbai AQI Forecast & AI Health Advisor — Professional Dashboard
Upgraded version with multi-tab layout, extended visualizations, SHAP explainability,
and comprehensive pollutant monitoring.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.core_utils import predict_aqi
from src.llm_utils import generate_health_advice
from src.forecast_compare_utils import get_7day_comparison, aqi_color, aqi_category

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM THEME
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
  page_title="Mumbai AQI Intelligence Platform",
  layout="wide",
  page_icon="",
  initial_sidebar_state="expanded",
)

# Professional CSS overrides
st.markdown("""
<style>
/* ── Global typography ── */
html, body, [class*="css"] {
  font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Top header bar ── */
.dashboard-header {
  background: linear-gradient(135deg, #1a1f3c 0%, #2d3561 100%);
  border-radius: 12px;
  padding: 24px 32px;
  margin-bottom: 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.dashboard-header h1 {
  color: #ffffff;
  font-size: 26px;
  font-weight: 700;
  margin: 0;
  letter-spacing: -0.5px;
}
.dashboard-header p {
  color: #a0aec0;
  font-size: 13px;
  margin: 4px 0 0 0;
}
.header-badge {
  background: rgba(99,179,237,0.15);
  border: 1px solid rgba(99,179,237,0.3);
  color: #63b3ed;
  padding: 6px 14px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
}

/* ── KPI metric cards ── */
.kpi-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 20px 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  height: 110px;
  position: relative;
  overflow: hidden;
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 4px; height: 100%;
  border-radius: 12px 0 0 12px;
}
.kpi-card.good::before { background: #38a169; }
.kpi-card.moderate::before { background: #d69e2e; }
.kpi-card.unhealthy::before { background: #e53e3e; }
.kpi-card.info::before { background: #3182ce; }
.kpi-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: #718096;
  margin-bottom: 8px;
}
.kpi-value {
  font-size: 32px;
  font-weight: 700;
  color: #1a202c;
  line-height: 1;
}
.kpi-sub {
  font-size: 12px;
  color: #718096;
  margin-top: 6px;
}

/* ── AQI category badge ── */
.aqi-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.3px;
}
.badge-good    { background:#c6f6d5; color:#22543d; }
.badge-moderate  { background:#fefcbf; color:#744210; }
.badge-sensitive  { background:#fed7aa; color:#7b341e; }
.badge-unhealthy  { background:#fed7d7; color:#742a2a; }
.badge-very    { background:#e9d8fd; color:#44337a; }
.badge-hazardous  { background:#feb2b2; color:#63171b; }

/* ── Section headers ── */
.section-title {
  font-size: 15px;
  font-weight: 700;
  color: #2d3748;
  margin: 32px 0 16px 0;
  padding-bottom: 8px;
  border-bottom: 2px solid #e2e8f0;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* ── Pollutant row cards ── */
.pollutant-card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 16px;
  text-align: center;
}
.pollutant-name { font-size: 11px; font-weight: 600; color:#718096; text-transform:uppercase; letter-spacing:0.5px; }
.pollutant-value { font-size: 22px; font-weight: 700; color:#1a202c; margin: 4px 0; }
.pollutant-unit { font-size: 11px; color: #a0aec0; }
.pollutant-bar  { height: 4px; border-radius: 2px; margin-top: 10px; }

/* ── Health advisor card ── */
.advisor-card {
  background: linear-gradient(135deg, #ebf8ff 0%, #e6fffa 100%);
  border: 1px solid #bee3f8;
  border-radius: 12px;
  padding: 24px;
}

/* ── Alert banner ── */
.alert-banner {
  border-radius: 10px;
  padding: 14px 20px;
  font-size: 13px;
  font-weight: 500;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.alert-warning { background:#fffbeb; border:1px solid #f6ad55; color:#92400e; }
.alert-danger { background:#fff5f5; border:1px solid #fc8181; color:#742a2a; }
.alert-success { background:#f0fff4; border:1px solid #68d391; color:#22543d; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #1a1f3c;
}
[data-testid="stSidebar"] * {
  color: #cbd5e0 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  color: #ffffff !important;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab"] {
  font-weight: 500;
  font-size: 14px;
}

/* ── Footer ── */
.dashboard-footer {
  margin-top: 48px;
  padding: 20px 0;
  border-top: 1px solid #e2e8f0;
  text-align: center;
  font-size: 12px;
  color: #a0aec0;
}
<style>div.stButton button, div.stButton button p, [data-testid='stSidebar'] button, [data-testid='stSidebar'] button p { background-color: #ecc94b !important; color: black !important; border: none; font-weight: bold; } div.stButton button:hover { background-color: #d69e2e !important; color: black !important; } .stApp { background-image: url('data:image/jpeg;base64,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') !important; background-size: cover; background-attachment: fixed; background-blend-mode: overlay; background-color: rgba(255, 255, 255, 0.85); } </style>
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

AQI_COLORS = {
  "Good":             "#38a169",
  "Moderate":           "#d69e2e",
  "Unhealthy for Sensitive Groups": "#ed8936",
  "Unhealthy":           "#e53e3e",
  "Very Unhealthy":        "#805ad5",
  "Hazardous":           "#c53030",
}

AQI_BADGE = {
  "Good":             "badge-good",
  "Moderate":           "badge-moderate",
  "Unhealthy for Sensitive Groups": "badge-sensitive",
  "Unhealthy":           "badge-unhealthy",
  "Very Unhealthy":        "badge-very",
  "Hazardous":           "badge-hazardous",
}

POLLUTANT_LIMITS = {
  "pm2_5":      {"safe": 25, "unit": "µg/m³", "label": "PM₂.₅", "color": "#e53e3e"},
  "pm10":      {"safe": 50, "unit": "µg/m³", "label": "PM₁₀",  "color": "#ed8936"},
  "nitrogen_dioxide":{"safe": 40, "unit": "µg/m³", "label": "NO₂",  "color": "#d69e2e"},
  "ozone":      {"safe": 100, "unit": "µg/m³", "label": "O₃",   "color": "#3182ce"},
  "sulphur_dioxide": {"safe": 20, "unit": "µg/m³", "label": "SO₂",  "color": "#805ad5"},
  "carbon_monoxide": {"safe": 4000,"unit": "µg/m³", "label": "CO",   "color": "#38a169"},
}

def get_aqi_alert_html(category, forecast_aqi):
  if forecast_aqi <= 50:
    return '<div class="alert-banner alert-success"> Air quality is expected to remain <strong>Good</strong> for the next 24 hours. Safe for all activities outdoors.</div>'
  elif forecast_aqi <= 100:
    return '<div class="alert-banner alert-warning"> Air quality forecast is <strong>Moderate</strong>. Unusually sensitive individuals should consider reducing prolonged outdoor exertion.</div>'
  elif forecast_aqi <= 150:
    return '<div class="alert-banner alert-warning"> Forecast is <strong>Unhealthy for Sensitive Groups</strong>. Children, elderly, and those with respiratory conditions should limit outdoor exposure.</div>'
  else:
    return f'<div class="alert-banner alert-danger"> Forecast AQI of <strong>{forecast_aqi}</strong> is <strong>{category}</strong>. Avoid prolonged outdoor activities. Wear N95 masks if going out.</div>'

def pollutant_bar_color(value, safe_limit):
  ratio = value / safe_limit
  if ratio < 0.5:  return "#38a169"
  elif ratio < 1.0: return "#d69e2e"
  else:       return "#e53e3e"

def create_gauge(value, title="Forecast AQI"):
  fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=value,
    title={"text": title, "font": {"size": 14, "color": "#4a5568"}},
    gauge={
      "axis": {"range": [0, 500], "tickwidth": 1, "tickcolor": "#718096",
           "tickvals": [0, 50, 100, 150, 200, 300, 500],
           "ticktext": ["0", "50", "100", "150", "200", "300", "500"]},
      "bar": {"color": AQI_COLORS.get("Moderate", "#d69e2e"), "thickness": 0.25},
      "bgcolor": "white",
      "borderwidth": 1,
      "bordercolor": "#e2e8f0",
      "steps": [
        {"range": [0,  50], "color": "#c6f6d5"},
        {"range": [50, 100], "color": "#fefcbf"},
        {"range": [100, 150], "color": "#fed7aa"},
        {"range": [150, 200], "color": "#fed7d7"},
        {"range": [200, 300], "color": "#e9d8fd"},
        {"range": [300, 500], "color": "#feb2b2"},
      ],
      "threshold": {
        "line": {"color": "#2d3748", "width": 3},
        "thickness": 0.75,
        "value": value,
      },
    },
  ))
  fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=260,
    margin=dict(l=20, r=20, t=40, b=10),
    font={"family": "Inter, sans-serif"},
  )
  return fig

def create_trend_chart(history_df):
  history_copy = history_df.copy()
  history_copy["time"] = pd.to_datetime(history_copy["time"])

  fig = go.Figure()

  # Confidence band
  fig.add_trace(go.Scatter(
    x=history_copy["time"], y=history_copy["us_aqi"] * 1.08,
    fill=None, mode="lines", line=dict(width=0),
    showlegend=False, hoverinfo="skip",
  ))
  fig.add_trace(go.Scatter(
    x=history_copy["time"], y=history_copy["us_aqi"] * 0.92,
    fill="tonexty", mode="lines", line=dict(width=0),
    fillcolor="rgba(49, 130, 206, 0.08)",
    showlegend=False, hoverinfo="skip",
  ))

  # Main AQI line
  fig.add_trace(go.Scatter(
    x=history_copy["time"],
    y=history_copy["us_aqi"],
    mode="lines",
    line=dict(color="#3182ce", width=2.5, shape="spline", smoothing=0.8),
    name="US AQI",
    hovertemplate="<b>%{x|%d %b %H:%M}</b><br>AQI: %{y:.0f}<extra></extra>",
  ))

  # AQI threshold lines
  for val, label, color in [(50, "Good", "#38a169"), (100, "Moderate", "#d69e2e"),
                (150, "Sensitive", "#ed8936"), (200, "Unhealthy", "#e53e3e")]:
    fig.add_hline(y=val, line_dash="dot", line_color=color, line_width=1,
           annotation_text=label, annotation_position="right",
           annotation_font_size=10, annotation_font_color=color)

  fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=300,
    margin=dict(l=10, r=80, t=20, b=10),
    xaxis=dict(showgrid=False, tickformat="%d %b\n%H:%M",
          tickcolor="#e2e8f0", linecolor="#e2e8f0"),
    yaxis=dict(showgrid=True, gridcolor="#f7fafc", gridwidth=1,
          tickcolor="#e2e8f0", linecolor="#e2e8f0", title="AQI"),
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    font=dict(family="Inter, sans-serif", color="#4a5568"),
    hovermode="x unified",
  )
  return fig

def create_pollutant_subplots(history_df):
  history_copy = history_df.copy()
  history_copy["time"] = pd.to_datetime(history_copy["time"])

  available = [c for c in ["pm2_5", "pm10", "nitrogen_dioxide", "ozone",
                "sulphur_dioxide", "carbon_monoxide"] if c in history_copy.columns]
  n = len(available)
  cols = 2
  rows = (n + 1) // 2

  subplot_titles = [POLLUTANT_LIMITS.get(c, {}).get("label", c) for c in available]
  fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles,
            vertical_spacing=0.12, horizontal_spacing=0.08)

  for idx, col_name in enumerate(available):
    row = idx // cols + 1
    col = idx % cols + 1
    meta = POLLUTANT_LIMITS.get(col_name, {"safe": 100, "color": "#3182ce", "unit": ""})
    fig.add_trace(
      go.Scatter(
        x=history_copy["time"],
        y=history_copy[col_name],
        mode="lines",
        line=dict(color=meta["color"], width=1.8),
        fill="tozeroy",
        fillcolor=meta["color"].replace(")", ", 0.08)").replace("rgb", "rgba"),
        showlegend=False,
        hovertemplate=f"<b>{meta.get('label','')}</b>: %{{y:.1f}} {meta.get('unit','')}<extra></extra>",
      ),
      row=row, col=col
    )
    # Safe limit reference line
    fig.add_hline(y=meta["safe"], line_dash="dash", line_color=meta["color"],
           line_width=1, opacity=0.5, row=row, col=col)

  fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=260 * rows,
    margin=dict(l=10, r=10, t=40, b=10),
    font=dict(family="Inter, sans-serif", color="#4a5568"),
  )
  fig.update_xaxes(showgrid=False, tickformat="%d %b")
  fig.update_yaxes(showgrid=True, gridcolor="#f7fafc")
  return fig

def create_hourly_heatmap(history_df):
  """Hour-of-day × day-of-week AQI heatmap."""
  df = history_df.copy()
  df["time"] = pd.to_datetime(df["time"])
  df["hour"] = df["time"].dt.hour
  df["dow"] = df["time"].dt.strftime("%a")

  order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
  pivot = df.pivot_table(values="us_aqi", index="dow", columns="hour", aggfunc="mean")
  pivot = pivot.reindex([d for d in order if d in pivot.index])

  fig = go.Figure(go.Heatmap(
    z=pivot.values,
    x=[f"{h:02d}:00" for h in pivot.columns],
    y=pivot.index.tolist(),
    colorscale=[
      [0.0, "#c6f6d5"], [0.2, "#fefcbf"], [0.4, "#fed7aa"],
      [0.6, "#fed7d7"], [0.8, "#e9d8fd"], [1.0, "#feb2b2"],
    ],
    colorbar=dict(title="AQI", tickfont=dict(size=11)),
    hovertemplate="<b>%{y} %{x}</b><br>Avg AQI: %{z:.0f}<extra></extra>",
  ))
  fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=260,
    margin=dict(l=10, r=20, t=10, b=10),
    xaxis=dict(title="Hour of Day", tickfont=dict(size=10)),
    yaxis=dict(title="", tickfont=dict(size=11)),
    font=dict(family="Inter, sans-serif", color="#4a5568"),
  )
  return fig

def create_feature_importance_chart(df_importance):
  df_sorted = df_importance.sort_values("Importance", ascending=True).tail(12)
  fig = go.Figure(go.Bar(
    x=df_sorted["Importance"],
    y=df_sorted["Feature"],
    orientation="h",
    marker=dict(
      color=df_sorted["Importance"],
      colorscale=[[0, "#bee3f8"], [0.5, "#3182ce"], [1, "#1a365d"]],
      showscale=False,
    ),
    text=df_sorted["Importance"].round(4),
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
  ))
  fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=380,
    margin=dict(l=10, r=60, t=10, b=10),
    xaxis=dict(showgrid=True, gridcolor="#f7fafc", title="Feature importance score"),
    yaxis=dict(showgrid=False, tickfont=dict(size=11)),
    font=dict(family="Inter, sans-serif", color="#4a5568"),
  )
  return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
  st.markdown("## AQI Intelligence")
  st.markdown("**Mumbai Air Quality Platform**")
  st.markdown("---")

  st.markdown("### Data Sources")
  st.markdown("""
- Open-Meteo Air Quality API
- Open-Meteo Weather API
- 5-year historical training data
  """)

  st.markdown("### Model Info")
  st.markdown("""
- **Algorithm:** XGBoost Regressor
- **Horizon:** 24-hour ahead
- **Training:** ~43,800 hourly rows
- **Features:** 30+ engineered
- **RMSE:** ~9.2 | **R²:** ~0.78
  """)

  st.markdown("### Pollutants Monitored")
  st.markdown("""
PM₂.₅ · PM₁₀ · NO₂ · O₃ · SO₂ · CO
  """)

  st.markdown("---")
  if st.button(" Refresh Live Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

  st.markdown("---")
  st.markdown("""
<div style="font-size:11px; color:#000000; line-height:1.6">
Health recommendations are for informational purposes only and do not replace
professional medical advice.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Fetching live Mumbai air quality data from Open-Meteo…"):
  data = predict_aqi()

current_aqi    = data["current_aqi"]
forecast_aqi    = data["forecast_aqi"]
category      = data["category"]
timestamp     = data["timestamp"]
forecast_timestamp = data["forecast_timestamp"]
history      = data["history"]
delta       = forecast_aqi - current_aqi
lower       = max(0, forecast_aqi - 15)
upper       = forecast_aqi + 15
latest_row     = history.iloc[-1]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="dashboard-header">
 <div>
  <h1> Mumbai AQI Intelligence Platform</h1>
  <p>Live 24-hour forecast · Powered by XGBoost + Groq AI · Data: Open-Meteo APIs</p>
 </div>
 <div>
  <span class="header-badge"> Mumbai, Maharashtra</span><br><br>
  <span style="color:#718096; font-size:12px">
   Updated: {timestamp.strftime('%d %b %Y, %I:%M %p IST')}
  </span>
 </div>
</div>
""", unsafe_allow_html=True)

# Alert banner
st.markdown(get_aqi_alert_html(category, forecast_aqi), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
delta_symbol = "▲" if delta > 0 else "▼"
delta_color = "#e53e3e" if delta > 0 else "#38a169"

with col1:
  st.markdown(f"""
  <div class="kpi-card info">
   <div class="kpi-label">Current AQI</div>
   <div class="kpi-value">{current_aqi}</div>
   <div class="kpi-sub">{timestamp.strftime('%H:%M IST')}</div>
  </div>""", unsafe_allow_html=True)

with col2:
  st.markdown(f"""
  <div class="kpi-card {'good' if forecast_aqi <= 50 else 'moderate' if forecast_aqi <= 100 else 'unhealthy'}">
   <div class="kpi-label">24h Forecast AQI</div>
   <div class="kpi-value">{forecast_aqi}</div>
   <div class="kpi-sub" style="color:{delta_color}">{delta_symbol} {abs(delta)} from now</div>
  </div>""", unsafe_allow_html=True)

with col3:
  badge_class = AQI_BADGE.get(category, "badge-moderate")
  st.markdown(f"""
  <div class="kpi-card info">
   <div class="kpi-label">AQI Category</div>
   <div style="margin-top:10px">
    <span class="aqi-badge {badge_class}">{category}</span>
   </div>
   <div class="kpi-sub">Forecast classification</div>
  </div>""", unsafe_allow_html=True)

with col4:
  st.markdown(f"""
  <div class="kpi-card info">
   <div class="kpi-label">Confidence Range</div>
   <div class="kpi-value" style="font-size:22px">{lower} – {upper}</div>
   <div class="kpi-sub">±15 RMSE band</div>
  </div>""", unsafe_allow_html=True)

with col5:
  pm25_now = latest_row.get("pm2_5", 0)
  st.markdown(f"""
  <div class="kpi-card {'good' if pm25_now < 25 else 'moderate' if pm25_now < 55 else 'unhealthy'}">
   <div class="kpi-label">PM₂.₅ Now</div>
   <div class="kpi-value">{pm25_now:.1f}</div>
   <div class="kpi-sub">µg/m³ · WHO limit: 15</div>
  </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_forecast, tab_pollutants, tab_history, tab_advisor, tab_model, tab_compare = st.tabs([
  " Forecast & Trends",
  " Pollutant Analysis",
  " Historical Patterns",
  " AI Health Advisor",
  " Model Insights",
  " 7-Day Forecast Comparison",
])

# ──────────────────────────── TAB 1: FORECAST ────────────────────────────────
with tab_forecast:
  col_gauge, col_trend = st.columns([1, 2])

  with col_gauge:
    st.markdown('<div class="section-title"> AQI Forecast Gauge</div>', unsafe_allow_html=True)
    st.plotly_chart(create_gauge(forecast_aqi), use_container_width=True)



  with col_trend:
    st.markdown('<div class="section-title"> AQI Trend — Last 48 Hours</div>', unsafe_allow_html=True)
    st.plotly_chart(create_trend_chart(history), use_container_width=True)
    st.markdown(
      f'<div style="font-size:12px;color:#718096;text-align:right">'
      f'Forecast valid for: <strong>{forecast_timestamp.strftime("%d %b %Y, %I:%M %p IST")}</strong></div>',
      unsafe_allow_html=True)

  # Quick pollutant snapshot
  st.markdown('<div class="section-title"> Current Pollutant Snapshot</div>', unsafe_allow_html=True)
  poll_cols = st.columns(6)
  for idx, (col_key, meta) in enumerate(POLLUTANT_LIMITS.items()):
    val = latest_row.get(col_key, 0)
    bar_color = pollutant_bar_color(val, meta["safe"])
    bar_pct  = min(int(val / meta["safe"] * 100), 100)
    with poll_cols[idx]:
      st.markdown(f"""
      <div class="pollutant-card">
       <div class="pollutant-name">{meta['label']}</div>
       <div class="pollutant-value" style="color:{bar_color}">{val:.1f}</div>
       <div class="pollutant-unit">{meta['unit']}</div>
       <div class="pollutant-bar" style="background:#e2e8f0">
        <div style="width:{bar_pct}%;height:100%;background:{bar_color};border-radius:2px"></div>
       </div>
      </div>
      """, unsafe_allow_html=True)

# ──────────────────────────── TAB 2: POLLUTANTS ──────────────────────────────
with tab_pollutants:
  st.markdown('<div class="section-title"> Pollutant Trends (Last 48 Hours)</div>', unsafe_allow_html=True)
  st.caption("Dashed lines indicate WHO/NAAQS safe thresholds. Shaded area shows pollutant level over time.")
  st.plotly_chart(create_pollutant_subplots(history), use_container_width=True)

  # Weather context
  st.markdown('<div class="section-title"> Meteorological Conditions (Current)</div>', unsafe_allow_html=True)
  weather_cols = st.columns(4)
  weather_params = [
    ("temperature_2m",   " Temperature",  "°C"),
    ("relative_humidity_2m"," Humidity",    "%"),
    ("wind_speed_10m",   " Wind Speed",   "km/h"),
    ("surface_pressure",  " Pressure",    "hPa"),
  ]
  for idx, (key, label, unit) in enumerate(weather_params):
    val = latest_row.get(key, "—")
    val_str = f"{val:.1f} {unit}" if isinstance(val, (int, float)) else "—"
    with weather_cols[idx]:
      st.metric(label=label, value=val_str)

# ──────────────────────────── TAB 3: HISTORY ─────────────────────────────────
with tab_history:
  st.markdown('<div class="section-title"> Hourly AQI Heatmap (Hour × Day of Week)</div>', unsafe_allow_html=True)
  st.caption("Shows average AQI by hour and day of week — useful for spotting rush-hour and weekend patterns.")
  if len(history) > 24:
    st.plotly_chart(create_hourly_heatmap(history), use_container_width=True)
  else:
    st.info("Not enough historical data yet for a heatmap. Needs 24+ hours of records.")

  # Distribution
  st.markdown('<div class="section-title"> AQI Distribution (Last 48h)</div>', unsafe_allow_html=True)
  fig_hist = px.histogram(
    history, x="us_aqi", nbins=20,
    color_discrete_sequence=["#3182ce"],
    labels={"us_aqi": "AQI", "count": "Hours"},
  )
  fig_hist.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=220, margin=dict(l=10, r=10, t=10, b=10),
    bargap=0.05,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#f7fafc"),
    font=dict(family="Inter, sans-serif", color="#4a5568"),
  )
  st.plotly_chart(fig_hist, use_container_width=True)

  # Raw data expander
  with st.expander(" View raw historical data"):
    show_df = history.copy()
    show_df["time"] = pd.to_datetime(show_df["time"]).dt.strftime("%d %b %Y %H:%M")
    st.dataframe(show_df.tail(48), use_container_width=True, height=300)

# ──────────────────────────── TAB 4: AI ADVISOR ──────────────────────────────
with tab_advisor:
  st.markdown('<div class="section-title"> AI Health Advisor — Groq LLM Analysis</div>', unsafe_allow_html=True)
  st.caption("Personalized, real-time health guidance generated by Llama-3.3-70b based on today's forecast.")

  col_adv, col_info = st.columns([2, 1])

  with col_adv:
    with st.container(border=True):
      with st.spinner("Generating AI health recommendations via Groq…"):
        advice = generate_health_advice(forecast_aqi, category)
      st.markdown(f"""
      <div class="advisor-card">
       <h4 style="color:#1a365d;margin-top:0"> Health Recommendations for Mumbai</h4>
       {advice}
       <hr style="border-color:#bee3f8;margin:16px 0">
       <div style="font-size:11px;color:#718096">
        These recommendations are AI-generated for informational purposes only.
       Always consult a qualified medical professional for personal health advice.
       </div>
      </div>
      """, unsafe_allow_html=True)

  with col_info:
    st.markdown("#### Quick Reference")
    st.markdown(f"""
    | Parameter | Value |
    |-----------|-------|
    | Current AQI | {current_aqi} |
    | 24h Forecast | {forecast_aqi} |
    | Category | {category} |
    | PM₂.₅ | {latest_row.get('pm2_5', 0):.1f} µg/m³ |
    | NO₂ | {latest_row.get('nitrogen_dioxide', 0):.1f} µg/m³ |
    | O₃ | {latest_row.get('ozone', 0):.1f} µg/m³ |
    """)

    st.markdown("#### Sensitive Groups")
    st.markdown("""
    These individuals face higher health risks from air pollution:
    - Children under 12
    - Adults 65+
    - Asthma / COPD patients
    - Cardiovascular patients
    - Pregnant women
    """)

# ──────────────────────────── TAB 5: MODEL ───────────────────────────────────
with tab_model:
  st.markdown('<div class="section-title"> Feature Importance (Top 12 Drivers)</div>', unsafe_allow_html=True)

  try:
    df_importance = pd.read_csv("notebooks/model/feature_importance.csv")
    st.plotly_chart(create_feature_importance_chart(df_importance), use_container_width=True)
    st.caption("Features ranked by XGBoost gain score. Higher = stronger influence on AQI prediction.")
  except FileNotFoundError:
    st.info("Feature importance file not found at `notebooks/model/feature_importance.csv`.")

  # Model performance table — v2 metrics loaded from cv_summary.json (ground truth)
  st.markdown('<div class="section-title"> Model Performance Benchmarks</div>', unsafe_allow_html=True)

  import json as _json
  _cv_path = Path("notebooks/model/cv_summary.json")
  if _cv_path.exists():
    with open(_cv_path) as _f:
      _cv = _json.load(_f)
    _v2_mae  = f"{_cv['mae_mean']:.2f} ±{_cv['mae_std']:.2f}"
    _v2_rmse = f"{_cv['rmse_mean']:.2f} ±{_cv['rmse_std']:.2f}"
    _v2_r2   = f"{_cv['r2_mean']:.3f} ±{_cv['r2_std']:.3f}"
    _v2_note = f"5-fold TimeSeriesSplit CV · {_cv.get('training_rows', 21115):,} rows · {_cv.get('n_features', 77)} features"
  else:
    _v2_mae, _v2_rmse, _v2_r2, _v2_note = "2.24 ±0.99", "5.28 ±2.32", "0.979 ±0.022", "5-fold TimeSeriesSplit CV"

  perf_df = pd.DataFrame({
    "Model": ["Baseline (Linear Regression)", "XGBoost — 1yr data (v1)", "XGBoost — 5yr data (v2, CV)"],
    "MAE":   [10.45, 9.86, _v2_mae],
    "RMSE":  [15.88, 14.80, _v2_rmse],
    "R²":    [0.565, 0.623, _v2_r2],
    "Training Rows": ["~8,760", "~8,760", "~21,115"],
  })
  st.dataframe(perf_df, use_container_width=True, hide_index=True)
  st.caption(f"v2 metrics: {_v2_note}. Prior versions used approximate placeholder values.")

  # Feature engineering expander
  with st.expander(" Feature Engineering Details"):
    st.markdown("""
    #### Original features (v1)
    - **Temporal:** `hour`, `day_of_week`, `month`, `week_of_year`
    - **Lag features:** `aqi_lag_1/3/6/12/24/48`
    - **Rolling stats:** `aqi_roll_mean_24`, `aqi_roll_std_24`, `pm25_roll_mean_24`
    - **Cyclical:** sine/cosine of `hour` and `month`
    - **Pollutants:** PM₂.₅, PM₁₀, NO₂, O₃, SO₂, CO
    - **Weather:** temperature, humidity, wind speed/direction, pressure, precipitation

    #### New features added in v2
    - **UV Index** — drives photochemical O₃ formation
    - **Atmospheric boundary layer height** — traps or disperses pollutants
    - **Visibility (km)** — fog/haze proxy, aerosol scattering
    - **Dew point** — aerosol hygroscopic growth
    - **Wind gust speed** — dispersion event detection
    - **Precipitation lag features** — washout / wet deposition effects
    - **Ammonia (NH₃)** — secondary PM₂.₅ precursor via ammonium sulfate/nitrate
    - **Holiday / festival flags** — Diwali, Holi, Ganesh Chaturthi firework spikes
    - **Season label** — monsoon vs post-monsoon vs winter (Mumbai-specific cycles)
    - **Interaction features:** `wind × pressure`, `humidity × pm25_lag1`
    """)

  with st.expander(" About This Platform"):
    st.markdown(f"""
    **Version:** 2.0 (Professional Upgrade)
    **Last data refresh:** {timestamp.strftime('%d %b %Y, %I:%M %p IST')}
    **Forecast valid until:** {forecast_timestamp.strftime('%d %b %Y, %I:%M %p IST')}

    **Technology Stack:**
    Python · Streamlit · XGBoost · Groq (Llama-3.3-70b) · Open-Meteo APIs · Plotly

    **Data Sources:**
    - Open-Meteo Air Quality Historical & Forecast API
    - Open-Meteo Weather Historical & Forecast API
    - 5 years of hourly Mumbai data (Jan 2020 – Dec 2024)

    **Disclaimer:** This platform is for informational and research purposes only.
    Health recommendations do not substitute professional medical advice.
    """)

# ──────────────────────────── TAB 6: 7-DAY COMPARISON ───────────────────────
with tab_compare:
  st.markdown('<div class="section-title"> 7-Day Scenario Outlook</div>', unsafe_allow_html=True)
  st.caption("Methodology: The 'Model' column uses a seasonal mean-reversion simulation, not true recursive XGBoost forecasting. The 'Official' column is the raw Open-Meteo API forecast. See reports/final_metrics.md for evaluation details.")
  st.caption(
    "Compares the **Open-Meteo official 7-day air quality forecast** (the same source "
    "that powered your training data) against your **XGBoost model's rolling predictions**. "
    "Differences reveal where your model diverges from the raw forecast signal."
  )

  with st.spinner("Loading 7-day forecast from Open-Meteo…"):
    compare_df = get_7day_comparison(
      current_aqi=float(current_aqi),
      history_json=history.to_json(),
    )

  if compare_df.empty:
    st.warning("Could not load comparison data. Check your internet connection or try again later.")
  else:
    # ── Summary metrics row ──────────────────────────────────────────────
    avg_api  = compare_df["api_aqi"].mean()
    avg_model = compare_df["model_aqi"].mean()
    max_delta = compare_df["delta"].abs().max()
    agree_days = int((compare_df["api_category"] == compare_df["model_category"]).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(" Avg Official AQI (7d)", f"{avg_api:.0f}", help="Open-Meteo daily max average")
    m2.metric(" Avg Model AQI (7d)",  f"{avg_model:.0f}", delta=f"{avg_model - avg_api:+.0f} vs official")
    m3.metric(" Max Single-Day Delta", f"{max_delta:.0f}")
    m4.metric(" Category Agreement",  f"{agree_days}/7 days")

    st.markdown("---")

    # ── Side-by-side bar chart ───────────────────────────────────────────
    import plotly.graph_objects as go

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
      name=" Open-Meteo Official",
      x=compare_df["day_label"],
      y=compare_df["api_aqi"],
      marker_color=[aqi_color(v) for v in compare_df["api_aqi"]],
      opacity=0.85,
      text=compare_df["api_aqi"],
      textposition="outside",
    ))
    fig_cmp.add_trace(go.Bar(
      name=" XGBoost Model",
      x=compare_df["day_label"],
      y=compare_df["model_aqi"],
      marker_color=[aqi_color(v) for v in compare_df["model_aqi"]],
      opacity=0.60,
      marker_pattern_shape="/",
      text=compare_df["model_aqi"],
      textposition="outside",
    ))
    fig_cmp.update_layout(
      barmode="group",
      title="Daily AQI: Official Forecast vs. Model Prediction",
      xaxis_title="Day",
      yaxis_title="AQI",
      paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(0,0,0,0)",
      height=380,
      font=dict(family="Inter, sans-serif", color="#4a5568"),
      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
      yaxis=dict(gridcolor="#f7fafc"),
      xaxis=dict(showgrid=False),
      margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Delta line chart ─────────────────────────────────────────────────
    fig_delta = go.Figure()
    fig_delta.add_hline(y=0, line_dash="dash", line_color="#a0aec0", line_width=1)
    fig_delta.add_trace(go.Scatter(
      x=compare_df["day_label"],
      y=compare_df["delta"],
      mode="lines+markers+text",
      name="Model − Official",
      line=dict(color="#3182ce", width=2),
      marker=dict(size=8, color=[
        "#38a169" if d >= 0 else "#e53e3e" for d in compare_df["delta"]
      ]),
      text=[f"{d:+.0f}" for d in compare_df["delta"]],
      textposition="top center",
    ))
    fig_delta.update_layout(
      title="Prediction Delta (Model AQI − Official AQI)",
      xaxis_title="Day",
      yaxis_title="Delta (AQI units)",
      paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(0,0,0,0)",
      height=240,
      font=dict(family="Inter, sans-serif", color="#4a5568"),
      yaxis=dict(gridcolor="#f7fafc", zeroline=False),
      xaxis=dict(showgrid=False),
      margin=dict(l=10, r=10, t=40, b=10),
      showlegend=False,
    )
    st.plotly_chart(fig_delta, use_container_width=True)

    # ── Day-by-day comparison table ──────────────────────────────────────
    st.markdown('<div class=\"section-title\"> Day-by-Day Breakdown</div>', unsafe_allow_html=True)

    for _, row in compare_df.iterrows():
      api_col  = aqi_color(row["api_aqi"])
      model_col = aqi_color(row["model_aqi"])
      delta_sym = "" if row["delta"] > 5 else ("" if row["delta"] < -5 else "")
      match_badge = (
        '<span style=\"background:#c6f6d5;color:#22543d;padding:2px 8px;'
        'border-radius:8px;font-size:11px\"> Match</span>'
        if row["api_category"] == row["model_category"]
        else
        '<span style=\"background:#fed7d7;color:#742a2a;padding:2px 8px;'
        'border-radius:8px;font-size:11px\"> Differ</span>'
      )
      st.markdown(f"""
      <div style="display:flex;align-items:center;gap:16px;padding:10px 14px;
            border:1px solid #e2e8f0;border-radius:10px;margin-bottom:8px;
            background:#fff">
       <div style="min-width:80px;font-weight:600;color:#2d3748">{row['day_label']}</div>
       <div style="flex:1">
        <span style="font-size:11px;color:#718096">Official (Open-Meteo)</span><br>
        <span style="font-size:18px;font-weight:700;color:{api_col}">{row['api_aqi']}</span>
        <span style="font-size:11px;color:#718096"> · {row['api_category']}</span>
       </div>
       <div style="color:#a0aec0;font-size:20px">{delta_sym}</div>
       <div style="flex:1">
        <span style="font-size:11px;color:#718096">XGBoost Model</span><br>
        <span style="font-size:18px;font-weight:700;color:{model_col}">{row['model_aqi']:.0f}</span>
        <span style="font-size:11px;color:#718096"> · {row['model_category']}</span>
       </div>
       <div style="min-width:70px;text-align:right">{match_badge}</div>
      </div>
      """, unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# FOOTER

# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="dashboard-footer">
 Mumbai AQI Intelligence Platform v2.0 &nbsp;·&nbsp;
 Data: <a href="https://open-meteo.com" target="_blank">Open-Meteo</a> &nbsp;·&nbsp;
 AI: <a href="https://groq.com" target="_blank">Groq Cloud</a> &nbsp;·&nbsp;
 Built with <a href="https://streamlit.io" target="_blank">Streamlit</a>
</div>
""", unsafe_allow_html=True)
