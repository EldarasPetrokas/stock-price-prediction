import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime

# FASTAPI ENDPOINT
API_URL = "http://127.0.0.1:8000/predict"

# STOCKS TO MONITOR
TECH_STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "TSLA", "META"]

st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

# 📌 HEADER
st.title("📈 Stock Prediction Dashboard")
st.write("This dashboard shows the **current stock price** and the **predicted closing price** for today.")

# DATE SELECTION
selected_date = st.date_input("📅 Select Prediction Date", datetime.date.today())

# SELECT STOCK
selected_stock = st.selectbox("📌 Choose a Stock", TECH_STOCKS)

# FETCH LIVE STOCK PRICE
def get_live_stock_price(ticker):
    stock = yf.Ticker(ticker)
    price = stock.history(period="1d")["Close"].iloc[-1]
    return round(price, 2)

# FETCH PREDICTION FROM FASTAPI
def get_predicted_price(ticker):
    try:
        response = requests.get(f"{API_URL}/{ticker}")
        if response.status_code == 200:
            return response.json().get("Predicted_Close_Price", "N/A")
        else:
            return "Prediction not available."
    except Exception as e:
        return f"Error: {str(e)}"

# DISPLAY RESULTS
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"📊 Current {selected_stock} Price")
    live_price = get_live_stock_price(selected_stock)
    st.metric(label="Live Price", value=f"${live_price}")

with col2:
    st.subheader(f"🔮 Predicted {selected_stock} Closing Price")
    predicted_price = get_predicted_price(selected_stock)
    if predicted_price:
        st.metric(label="Predicted Close", value=f"${predicted_price}")
    else:
        st.warning("⚠️ Prediction not available.")

# HISTORICAL STOCK DATA & CHART
st.subheader(f"📉 {selected_stock} Historical Prices")
df = yf.download(selected_stock, period="6mo")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["Close"], label="Close Price", color="blue")
ax.set_title(f"{selected_stock} Price Chart (Last 6 Months)")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price ($)")
ax.legend()
st.pyplot(fig)
