import yfinance as yf
import pandas as pd
import os

# Get the absolute path to the root project directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Ensure the 'data/' directory exists in the root folder
os.makedirs(DATA_DIR, exist_ok=True)


def get_stock_data(tickers, start="2010-01-01", end="2025-03-13"):
    # Create an empty DataFrame to store all stocks data
    all_stocks_data = []

    for ticker in tickers:
        # Download data for each stock
        stock = yf.download(ticker, start=start, end=end)
        stock["Ticker"] = ticker  # Add a column to identify the stock

        # Reset index to ensure 'Date' is a column (not the index)
        stock.reset_index(inplace=True)

        # Convert 'Date' column to datetime format
        stock['Date'] = pd.to_datetime(stock['Date'], errors='coerce')

        # Check if 'Date' is not NaT and drop rows with invalid Date
        stock = stock[stock['Date'].notna()]

        # Save each stock's data as a separate CSV file
        file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        stock.to_csv(file_path, index=False)  # Save without 'Date' as the index

        # Append the stock data to the list
        all_stocks_data.append(stock)

    # Concatenate all stock data into a single DataFrame
    combined_data = pd.concat(all_stocks_data)

    # Set 'Date' as the index after concatenating
    combined_data.set_index("Date", inplace=True)

    return combined_data


if __name__ == "__main__":
    # List of tech stock tickers
    tech_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "TSLA", "META"]

    # Get data for the list of tech stocks
    combined_data = get_stock_data(tech_stocks)

    # Print out the first few rows of the combined dataset
    print(combined_data.head())
