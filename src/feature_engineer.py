import pandas as pd
import os
import ta

# Get the absolute path to the root project directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def load_stock_data(ticker):
    """ Load stock data from CSV and fix column names """
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")

    # Load CSV and skip the first two metadata rows
    df = pd.read_csv(file_path, skiprows=2)

    # Rename the first column to 'Date' if it's not correctly labeled
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # Check if the correct column names exist, otherwise fix them
    expected_columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    actual_columns = df.columns.tolist()

    if not set(expected_columns).issubset(actual_columns):
        print(f"⚠️ Warning: Columns may be misaligned. Found columns: {actual_columns}")

        # Try to rename columns based on index if needed
        df.columns = ["Date", "Close", "High", "Low", "Open", "Volume", "Ticker"]

    # Convert 'Date' column to datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Set it as the index
    df.set_index("Date", inplace=True)

    return df


def add_technical_indicators(df):
    """ Add Moving Averages, RSI, MACD, Bollinger Bands, and ATR to DataFrame """

    # Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD and Signal Line
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
    df['Upper_Band'] = bollinger.bollinger_hband()
    df['Lower_Band'] = bollinger.bollinger_lband()

    # Average True Range (ATR) - Measures Volatility
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()

    # Drop NaN values caused by rolling calculations
    df.dropna(inplace=True)

    return df


def save_processed_data(df, ticker):
    """ Save processed stock data with indicators """
    processed_file = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
    df.to_csv(processed_file)
    print(f"✅ Processed data saved: {processed_file}")


def process_multiple_stocks(tickers):
    """ Process multiple stocks and save their data """
    for ticker in tickers:
        print(f"Processing data for {ticker}...")
        df = load_stock_data(ticker)
        df = add_technical_indicators(df)
        save_processed_data(df, ticker)


if __name__ == "__main__":
    # List of tech stock tickers
    tech_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "TSLA", "META"]

    # Process the list of stocks
    process_multiple_stocks(tech_stocks)
