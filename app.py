
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta

# ========== CONFIG ==========
st.set_page_config(page_title="ETH Forecast Dashboard", layout="wide")
st.title("📊 Ethereum (ETH) Price Dashboard")

# Sidebar config
st.sidebar.header("📍 Forecast Settings")
mu = st.sidebar.slider("Expected Return (μ)", 0.0, 0.02, 0.01, step=0.001)
sigma = st.sidebar.slider("Volatility (σ)", 0.0, 0.05, 0.01, step=0.001)

# ========== LOAD LIVE DATA ==========
@st.cache_data(ttl=60)
def fetch_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "ETHUSDT", "interval": "1h", "limit": 100}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades", "taker_base_vol", "taker_quote_vol", "ignore"
    ])
    df["time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    df.set_index("time", inplace=True)
    return df

df = fetch_data()

# ========== INDICATORS ==========
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

rsi = calculate_rsi(df["close"])
macd, signal = calculate_macd(df["close"])

# ========== PRICE ZONES ==========
last_price = df["close"].iloc[-1]
discount_zone = df["close"].quantile(0.25)
premium_zone = df["close"].quantile(0.75)

# ========== GBM FORECAST ==========
def gbm_forecast(S0, mu, sigma, steps=24):
    dt = 1 / 24
    forecast = [S0]
    for _ in range(steps):
        dS = forecast[-1] * (mu * dt + sigma * np.random.normal() * np.sqrt(dt))
        forecast.append(forecast[-1] + dS)
    return forecast

forecast = gbm_forecast(last_price, mu, sigma)

# ========== LAYOUT ==========
col1, col2 = st.columns([3, 1])

# -- CHART MAIN --
with col1:
    st.markdown("### 🟢 Live ETH/USDT Chart (24H - 1H Candles)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Price"))
    fig.update_layout(height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# -- STATS & SIGNALS --
with col2:
    st.markdown("### 🧾 Price Info")
    st.metric("Last Price", f"${last_price:,.2f}")
    change = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100
    st.metric("24H Change", f"{change:.2f}%")

    st.markdown("### 🧠 GBM Forecast (24h)")
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(y=forecast, mode="lines", name="Forecast Price"))
    forecast_fig.update_layout(height=250, margin=dict(t=10, b=10))
    st.plotly_chart(forecast_fig, use_container_width=True)

# ========== ALERTS ==========
st.markdown("### 🔔 Indicator Alerts")
rsi_value = rsi.iloc[-1]
macd_diff = macd.iloc[-1] - signal.iloc[-1]
macd_prev_diff = macd.iloc[-2] - signal.iloc[-2]

if rsi_value < 30:
    st.success("🟢 RSI Alert: BUY (Oversold)")
elif rsi_value > 70:
    st.error("🔴 RSI Alert: SELL (Overbought)")
else:
    st.info("⚪ RSI: Neutral")

if macd_diff > 0 and macd_prev_diff < 0:
    st.success("🟢 MACD Bullish Crossover")
elif macd_diff < 0 and macd_prev_diff > 0:
    st.error("🔴 MACD Bearish Crossover")
else:
    st.info("⚪ MACD: No Crossover")

if last_price < discount_zone:
    st.success("🟢 Price in DISCOUNT zone — potential BUY")
elif last_price > premium_zone:
    st.error("🔴 Price in PREMIUM zone — caution")
else:
    st.info("⚪ Price in FAIR VALUE zone")

# ========== FOOTER ==========
st.caption("🔁 Auto-updates every 1 minute — Powered by Binance API + Streamlit + Plotly.")
