import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
import requests
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, IchimokuIndicator
from ta.volatility import BollingerBands

# --- CONFIG ---
st.set_page_config(page_title="Crypto Dashboard", layout="wide")
symbols = {"ETH/USDT": "ETHUSDT", "BTC/USDT": "BTCUSDT", "SOL/USDT": "SOLUSDT"}
symbol = st.sidebar.selectbox("Choose Symbol", list(symbols.keys()))
symbol_binance = symbols[symbol]

mu = st.sidebar.slider("Expected Return (μ)", 0.0000, 0.0200, 0.0100)
sigma = st.sidebar.slider("Volatility (σ)", 0.000, 0.100, 0.060)
phi = st.sidebar.slider("Phi (Drift Adj)", 0.90, 1.10, 1.00)
lambda_ = st.sidebar.slider("Lambda (Rate)", 0.1, 2.0, 1.0)

# --- DATA FETCH ---
@st.cache_data(ttl=3600)
def get_data(symbol):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=200"
    try:
        raw = requests.get(url, timeout=10)
        raw.raise_for_status()
        data = raw.json()
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume", "", "", "", "", "", ""])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        st.error(f"API Error: {e}")
        return pd.DataFrame()

df = get_data(symbol_binance)

if df.empty or "close" not in df.columns:
    st.error("🚫 Failed to fetch price data. Check your symbol or Binance API connection.")
    st.stop()

latest_price = df["close"].iloc[-1]

# --- INDICATORS ---
df["RSI"] = RSIIndicator(df["close"]).rsi()
df["MACD_diff"] = MACD(df["close"]).macd_diff()
df["EMA50"] = EMAIndicator(df["close"], window=50).ema_indicator()
df["EMA200"] = EMAIndicator(df["close"], window=200).ema_indicator()
bb = BollingerBands(df["close"])
df["BB_upper"] = bb.bollinger_hband()
df["BB_lower"] = bb.bollinger_lband()
ichi = IchimokuIndicator(df["high"], df["low"])
df["Cloud_Lead1"] = ichi.ichimoku_a()
df["Cloud_Lead2"] = ichi.ichimoku_b()

premium = df["close"].rolling(24).max()
discount = df["close"].rolling(24).min()

# --- MAIN DASH ---
st.title(f"{symbol} Dashboard")
col1, col2 = st.columns([3, 1])
with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=premium, name="Premium", line=dict(dash="dot", color="green")))
    fig.add_trace(go.Scatter(x=df.index, y=discount, name="Discount", line=dict(dash="dot", color="red")))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Price Info")
    st.metric("Last Price", f"${latest_price:,.2f}")
    st.metric("24H Change", f"{(df['close'].pct_change().iloc[-1] * 100):.2f}%")
    st.write("RSI:", round(df["RSI"].iloc[-1], 2))
    st.write("MACD:", "Bullish" if df["MACD_diff"].iloc[-1] > 0 else "Bearish")
    st.write("EMA Cross:", "Bullish" if df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1] else "Bearish")

# --- ALERTS ---
st.subheader("⚠ Indicator Alerts")
if df["RSI"].iloc[-1] > 70:
    st.error("RSI: Overbought (SELL)")
elif df["RSI"].iloc[-1] < 30:
    st.success("RSI: Oversold (BUY)")
else:
    st.info("RSI: Neutral")

# --- GBM 24H FORECAST (enhanced) ---
st.subheader("🔮 GBM Forecast (24H)")
def custom_forecast(S0, mu, sigma, t, phi, lambda_):
    drift = (mu - 0.5 * sigma ** 2) * t
    shock = sigma * np.sqrt(t) * phi
    adjusted = S0 * np.exp((drift + shock) * lambda_)
    return adjusted

expected_price = custom_forecast(latest_price, mu, sigma, 1, phi, lambda_)
st.write(f"*Expected price in 24H:* ${expected_price:,.2f}")

std_dev = latest_price * (np.sqrt(np.exp((sigma**2)*1) - 1))
low = round(expected_price - std_dev, 2)
high = round(expected_price + std_dev, 2)
st.write(f"*Confidence range (68%):* ${low} to ${high}")

# --- PER-HOUR FORECAST ---
st.subheader("🕒 GBM Per-Hour Forecast")
hours = np.arange(1, 25)
hourly_forecast = [custom_forecast(latest_price, mu, sigma, t/24, phi, lambda_) for t in hours]
hourly_df = pd.DataFrame({"Hour": hours, "Forecast": hourly_forecast})

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=hourly_df["Hour"], y=hourly_df["Forecast"], name="Forecast", line=dict(color="purple")))
fig2.update_layout(title="Hourly GBM Forecast", xaxis_title="Hour", yaxis_title="Price (USD)")
st.plotly_chart(fig2, use_container_width=True)

# --- BACKTEST SIGNAL ---
st.subheader("📈 Backtest Signal (RSI/MACD)")
df["Buy"] = (df["RSI"] < 30) & (df["MACD_diff"] > 0)
df["Sell"] = (df["RSI"] > 70) & (df["MACD_diff"] < 0)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close"))
fig3.add_trace(go.Scatter(x=df[df["Buy"]].index, y=df[df["Buy"]]["close"], mode="markers", name="Buy", marker=dict(color="green", size=8)))
fig3.add_trace(go.Scatter(x=df[df["Sell"]].index, y=df[df["Sell"]]["close"], mode="markers", name="Sell", marker=dict(color="red", size=8)))
fig3.update_layout(title="Backtested Buy/Sell Signals", height=500)
st.plotly_chart(fig3, use_container_width=True)

