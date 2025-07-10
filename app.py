import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator

# Custom Forecast Function
def gbm_forecast(S0, mu, sigma, phi, lambd, horizon=24):
    dt = 1 / 24
    forecast = [S0]
    for _ in range(horizon):
        drift = (mu - 0.5 * sigma**2) * dt * phi
        shock = sigma * np.sqrt(dt) * np.random.normal() * lambd
        S_next = forecast[-1] * np.exp(drift + shock)
        forecast.append(S_next)
    return forecast

# Symbol map
symbol_map = {
    "ETH-USD": "ethusdt",
    "BTC-USD": "btcusdt",
    "SOL-USD": "solusdt"
}

# App UI
st.sidebar.header("📍 Forecast Settings")
symbol = st.sidebar.selectbox("Choose Symbol", list(symbol_map.keys()))
mu = st.sidebar.slider("Expected Return (μ)", 0.00, 0.02, 0.01, 0.001)
sigma = st.sidebar.slider("Volatility (σ)", 0.00, 0.10, 0.06, 0.001)
phi = st.sidebar.slider("Phi (Drift Adj)", 0.90, 1.10, 1.00, 0.01)
lambd = st.sidebar.slider("Lambda (Rate)", 0.10, 2.00, 1.00, 0.01)

st.title("📊 Ethereum (ETH) Price Dashboard")

def fetch_binance_klines(symbol, interval='1h', limit=500):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Open time', inplace=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

try:
    df = fetch_binance_klines(symbol_map[symbol])
    if df.empty or len(df) < 25:
        st.error("⚠️ Not enough data to compute indicators.")
    else:
        # Indicators
        df['EMA50'] = EMAIndicator(df['Close'], window=50).ema_indicator()
        df['EMA200'] = EMAIndicator(df['Close'], window=200).ema_indicator()
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        bollinger = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        df['Premium'] = df['Close'].rolling(24).max() - df['Close']
        df['Discount'] = df['Close'] - df['Close'].rolling(24).min()

        last_price = df['Close'].iloc[-1]
        forecast = gbm_forecast(last_price, mu, sigma, phi, lambd)

        st.subheader(f"{symbol} Latest Price: ${last_price:,.2f}")
        st.line_chart(pd.Series(forecast, name="24H Forecast"))

        st.subheader("📈 Technical Snapshot")
        st.write(df.tail(5))
except Exception as e:
    st.error(f"Something went wrong: {e}")
