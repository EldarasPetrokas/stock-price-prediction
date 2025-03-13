import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Get absolute paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")

# Ensure the plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

# List of tech stock tickers to evaluate
TECH_STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "TSLA", "META"]


def load_processed_data(ticker):
    """ Load processed stock data with indicators """
    file_path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Select Features for LSTM (Must be Scaled)
    features = ["SMA_50", "SMA_200", "EMA_12", "EMA_26", "MACD", "RSI", "Upper_Band", "Lower_Band", "ATR"]
    target = "Close"

    # Scale data between 0 and 1
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features + [target]])

    return df, df_scaled, scaler


def create_sequences(data, sequence_length=50):
    """ Convert data into sequences for LSTM """
    X, y = [], []
    for i in range(len(data) - sequence_length - 1):
        X.append(data[i: i + sequence_length, :-1])  # Features
        y.append(data[i + sequence_length, -1])  # Target (Close Price)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def evaluate_lstm_model(ticker):
    """ Evaluate the performance of the trained LSTM model """
    df, df_scaled, scaler = load_processed_data(ticker)

    sequence_length = 100
    X, y = create_sequences(df_scaled, sequence_length)

    # Split into training and testing sets
    split = int(0.8 * len(X))  # 80% Train, 20% Test
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Load the trained LSTM model
    model_path = os.path.join(MODEL_DIR, "lstm_multi_stock.keras")
    model = tf.keras.models.load_model(model_path)

    # Make predictions
    predictions = model.predict(X_test)

    # Inverse scale the target variable (Close Price)
    y_test_rescaled = scaler.inverse_transform(
        np.column_stack((np.zeros((y_test.shape[0], df_scaled.shape[1] - 1)), y_test)))[:, -1]
    predictions_rescaled = scaler.inverse_transform(
        np.column_stack((np.zeros((predictions.shape[0], df_scaled.shape[1] - 1)), predictions)))[:, -1]

    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    r2 = r2_score(y_test_rescaled, predictions_rescaled)

    print(f"✅ {ticker} Model Performance:")
    print(f"📊 RMSE: {rmse:.2f}")
    print(f"📊 R² Score: {r2:.2f}\n")

    # Get corresponding dates for test set
    test_dates = df.index[-len(y_test_rescaled):]

    # Plot Actual vs Predicted Prices
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_rescaled, label="Actual Prices", color="blue")
    plt.plot(test_dates, predictions_rescaled, label="Predicted Prices", color="red", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"LSTM Stock Price Prediction for {ticker}")
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Show every month
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()

    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, f"{ticker}_prediction.png")
    plt.savefig(plot_path)
    print(f"📈 Prediction plot saved: {plot_path}\n")
    plt.show()


if __name__ == "__main__":
    for stock in TECH_STOCKS:
        evaluate_lstm_model(stock)
