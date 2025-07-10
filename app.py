import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
from datetime import datetime, timedelta

# === CONFIG ===
st.set_page_config(page_title="ETH Forecast Dashboard", layout="wide")

# === PARAMETERS ===
st.sidebar.markdown("🎯 **Forecast Settings**")
mu = st.sidebar.slider("Expected Return (μ)", 0.00, 0.02, 0.01, step=0.001)
sigma = st.sidebar.slider("Volatility (σ)", 0.00, 0.10, 0.06, step=0.001)
phi = st.sidebar.slider("Phi (Drift Adj)", 0.90, 1.10, 1.0, step=0.01)
lamb = st.sidebar.slider("Lambda (Rate)", 0.1, 2.0, 1.0, step=0.1)

# === FUNCTIONS ===

def get_binance_data(symbol="ETHUSDT", interval="1h", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["close"] = df["close"].astype(float)
    df.set_index("timestamp", inplace=True)
    return df[["close"]]

def gbm_forecast(S0, mu, sigma, phi, lamb, T=24, dt=1):
    steps = int(T / dt)
    prices = [S0]
    for _ in range(steps):
        drift = (mu - 0.5 * sigma ** 2) * dt
        shock = sigma * np.sqrt(dt) * phi * lamb * np.random.normal()
        S_t = prices[-1] * np.exp(drift + shock)
        prices.append(S_t)
    return prices

# === FETCH LIVE DATA ===
df = get_binance_data()

if df.empty or len(df) < 24:
    st.error("Live data fetch failed or too short. Try again later.")
else:
    # === FORECAST CALC ===
    last_price = df["close"].iloc[-1]
    forecast = gbm_forecast(S0=last_price, mu=mu, sigma=sigma, phi=phi, lamb=lamb)
    forecast_times = [df.index[-1] + timedelta(hours=i) for i in range(len(forecast))]

    # === PREMIUM/DISCOUNT ZONES ===
    premium = df["close"].rolling(24).max().iloc[-1]
    discount = df["close"].rolling(24).min().iloc[-1]

    # === PLOT ===
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode='lines', name='ETH Price'))
    fig.add_trace(go.Scatter(x=forecast_times, y=forecast, mode='lines', name='Forecast'))

    fig.add_hline(y=premium, line_dash="dot", line_color="red", annotation_text="Premium Zone", annotation_position="top left")
    fig.add_hline(y=discount, line_dash="dot", line_color="green", annotation_text="Discount Zone", annotation_position="bottom left")

    fig.update_layout(title="📈 Ethereum (ETH) Price Dashboard", xaxis_title="Time", yaxis_title="Price (USD)", template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

    # === STATS ===
    st.subheader("🔍 Latest Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Last Price", f"${last_price:,.2f}")
    col2.metric("Premium", f"${premium:,.2f}")
    col3.metric("Discount", f"${discount:,.2f}")
