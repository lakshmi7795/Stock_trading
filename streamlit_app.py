import streamlit as st
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
import requests

st.set_page_config(page_title="📊 AI Stock Monitor", layout="centered")
st.title("📈 AI Stock Signal Predictor")

TOKEN = "YOUR_TELEGRAM_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"
REFRESH_INTERVAL = 3600

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

time_since = time.time() - st.session_state.last_refresh
time_remaining = int(REFRESH_INTERVAL - time_since)
if time_remaining > 0:
    mins, secs = divmod(time_remaining, 60)
    st.info(f"🔄 Auto-refresh in: {mins} min {secs} sec")
    time.sleep(1)

stock = st.selectbox("Select a Stock", ["TCS.NS", "RELIANCE.NS", "BANKBARODA.NS"])
df = yf.download(stock, period="7d", interval="30m", progress=False)

if not df.empty:
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()
    df['macd'] = ta.trend.MACD(close=df['Close']).macd()
    df['ema_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df.dropna(inplace=True)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    X = df[['rsi','macd','ema_20']]
    y = df['target']
    model = RandomForestClassifier()
    model.fit(X, y)

    latest = df.iloc[-1][['rsi','macd','ema_20']].values.reshape(1, -1)
    signal = None
    if np.any(np.isnan(latest)) or np.any(np.isinf(latest)):
        st.warning("⚠️ Not enough data")
    else:
        pred = model.predict(latest)[0]
        signal = "BUY 🟢" if pred==1 else "WAIT 🔴"
        st.metric(label="📢 Signal", value=signal)

    if signal=="BUY 🟢" and st.session_state.last_refresh <= time.time()-REFRESH_INTERVAL:
        msg = f"📢 BUY signal for {stock}"
        requests.get(f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={msg}")
        st.success("✅ Telegram alert sent.")

    st.subheader(f"{stock} Price Chart")
    st.line_chart(df['Close'])
else:
    st.error("❌ Fetching data failed")

if time_remaining <= 0:
    st.session_state.last_refresh = time.time()
    st.experimental_rerun()
