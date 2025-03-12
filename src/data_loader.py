import yfinance as yf
import pandas as pd
import os

# Get the absolute path to the root project directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Ensure the 'data/' directory exists in the root folder
os.makedirs(DATA_DIR, exist_ok=True)


def get_stock_data(ticker, start="2020-01-01", end="2025-03-11"):
    stock = yf.download(ticker, start=start, end=end)

    # Save CSV to the correct 'data/' folder
    stock.to_csv(os.path.join(DATA_DIR, f"{ticker}.csv"))
    return stock


if __name__ == "__main__":
    df = get_stock_data("AAPL")
    print(df.head())  # Show first few rows
