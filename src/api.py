import os
import uvicorn
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import MinMaxScaler

# Define paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

# Load trained LSTM model
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_multi_stock.keras")
lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)

# Define supported stock tickers
TECH_STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "TSLA", "META"]

# Initialize FastAPI app
app = FastAPI(title="Stock Price Prediction API", version="1.0")


# Function to load latest stock data
def load_stock_data(ticker):
    file_path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Select features used during training
    features = ["SMA_50", "SMA_200", "EMA_12", "EMA_26", "MACD", "RSI", "Upper_Band", "Lower_Band", "ATR"]

    # Scale data
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features + ["Close"]])

    # Select last 50 days for LSTM input
    sequence_length = 50
    X_input = df_scaled[-sequence_length:, :-1]  # Exclude target "Close"

    return np.array([X_input], dtype=np.float32), scaler, df.index[-1]


# API endpoint for stock price prediction
@app.get("/predict/{ticker}")
def predict_stock(ticker: str):
    if ticker not in TECH_STOCKS:
        return {"error": "Ticker not supported. Choose from: " + ", ".join(TECH_STOCKS)}

    # Load and preprocess data
    X_input, scaler, last_date = load_stock_data(ticker)

    # Make prediction
    predicted_price_scaled = lstm_model.predict(X_input)[0][0]

    # Inverse scale prediction
    predicted_price = scaler.inverse_transform(
        np.column_stack((np.zeros((1, X_input.shape[2])), [predicted_price_scaled]))
    )[:, -1][0]

    return {
        "Ticker": ticker,
        "Last_Data_Date": str(last_date),
        "Predicted_Close_Price": round(predicted_price, 2)
    }


# Run FastAPI using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
