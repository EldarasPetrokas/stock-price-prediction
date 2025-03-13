import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

# Enable Mixed Precision for Apple Silicon (M1/M2/M3)
if tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Get absolute paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# List of tech stock tickers
TECH_STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "TSLA", "META"]


def load_multiple_stocks(tickers):
    """ Load and scale multiple stock datasets """
    all_data = []
    scalers = {}

    for ticker in tickers:
        file_path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
        df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

        features = ["SMA_50", "SMA_200", "EMA_12", "EMA_26", "MACD", "RSI", "Upper_Band", "Lower_Band", "ATR"]
        target = "Close"

        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df[features + [target]])
        scalers[ticker] = scaler  # Store scaler for each stock

        all_data.append(df_scaled)

    combined_data = np.concatenate(all_data, axis=0)
    return combined_data, scalers


def create_sequences(data, sequence_length=50):
    """ Convert data into sequences for LSTM """
    X, y = [], []
    for i in range(len(data) - sequence_length - 1):
        X.append(data[i: i + sequence_length, :-1])  # Features
        y.append(data[i + sequence_length, -1])  # Target (Close Price)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)  # Convert to float32 for Metal


def train_lstm_model():
    """ Train an LSTM Model using all tech stocks """
    sequence_length = 100

    # Load and combine data for multiple stocks
    df_scaled, scalers = load_multiple_stocks(TECH_STOCKS)

    # Prepare training data
    X, y = create_sequences(df_scaled, sequence_length)

    # Split into training and testing sets
    split = int(0.8 * len(X))  # 80% Train, 20% Test
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM Model (Bidirectional LSTM, Dropout, etc.)
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=(sequence_length, X.shape[2]), dtype="float32")),
        Dropout(0.3),
        LSTM(64, dtype="float32"),
        Dropout(0.3),
        Dense(1, dtype="float32")  # Predicting stock price
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    # Early Stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # Train the model with early stopping
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Save in Keras format (instead of legacy HDF5 format)
    model.save(os.path.join(MODEL_DIR, "lstm_multi_stock.keras"), save_format="keras")
    print("✅ Multi-Stock LSTM Model trained and saved!")


if __name__ == "__main__":
    train_lstm_model()