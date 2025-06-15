import streamlit as st
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import time
import requests

# --- Streamlit UI setup
st.set_page_config(page_title="AI Stock Monitor", layout="centered")
st.title("AI Stock Signal Predictor")

# --- Telegram Bot Config
TOKEN = "7647392071:AAGjFGHdFd5pSfBLwgKp9iFsL0O94u2kDZY"
CHAT_ID = "5570374030"
REFRESH_INTERVAL = 3600  # 1 hour in seconds

# --- Auto-refresh tracker
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

time_since = time.time() - st.session_state.last_refresh
time_remaining = int(REFRESH_INTERVAL - time_since)
if time_remaining > 0:
    mins, secs = divmod(time_remaining, 60)
    st.info(f"⏱️ Auto-refresh in: {mins} min {secs} sec")
    time.sleep(1)

# --- Stock selection
stock = st.selectbox("Select a Stock", ["TCS.NS", "RELIANCE.NS", "BANKBARODA.NS"])
df = yf.download(stock, period="7d", interval="30m", progress=False)

# --- Check if data is returned
if not df.empty:
    # ✅ Ensure df['Close'] is a proper 1D Series
    if isinstance(df['Close'], pd.DataFrame):
        close_prices = df['Close'].iloc[:, 0]
    else:
        close_prices = df['Close']

    # --- Add technical indicators
    df['rsi'] = ta.momentum.RSIIndicator(close=close_prices).rsi()
    df['macd'] = ta.trend.MACD(close=close_prices).macd()
    df['ema_20'] = ta.trend.EMAIndicator(close=close_prices, window=20).ema_indicator()
    df.dropna(inplace=True)

    # --- AI model
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    X = df[['rsi', 'macd', 'ema_20']]
    y = df['target']
    model = RandomForestClassifier()
    model.fit(X, y)

    # --- Predict on latest row
    latest = df.iloc[-1][['rsi', 'macd', 'ema_20']].values.reshape(1, -1)
    if np.any(np.isnan(latest)) or np.any(np.isinf(latest)):
        st.warning("⚠️ Not enough clean data to predict.")
    else:
        pred = model.predict(latest)[0]
        signal = "BUY" if pred == 1 else "WAIT"
        st.metric(label="📈 AI Signal", value=signal)

        # --- Telegram alert
        if signal == "BUY" and st.session_state.last_refresh <= time.time() - REFRESH_INTERVAL:
            msg = f"📢 BUY Alert: {stock} is showing a BUY signal"
            url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={msg}"
            requests.get(url)
            st.success("✅ Telegram alert sent!")

    # --- Chart display
    st.subheader(f"{stock} Price Chart")
    st.line_chart(df['Close'])

else:
    st.error("❌ Failed to fetch stock data. Try another stock or check your connection.")

# --- Auto-rerun after interval
if time_remaining <= 0:
    st.session_state.last_refresh = time.time()
    st.experimental_rerun()
