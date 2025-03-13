import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Get absolute paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


def load_processed_data(ticker):
    """ Load processed stock data with indicators """
    file_path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    return df


def train_model(df):
    """ Train a Random Forest Regression model to predict stock prices """

    # Select Features (Technical Indicators)
    features = ["SMA_50", "SMA_200", "EMA_12", "EMA_26", "MACD", "RSI", "Upper_Band", "Lower_Band", "ATR"]
    target = "Close"

    # Define Input (X) and Output (y)
    X = df[features]
    y = df[target].shift(-1)  # Predict next day's closing price

    # Drop last row (NaN target after shifting)
    X, y = X.iloc[:-1], y.iloc[:-1]

    # Split into Training and Test sets (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make Predictions
    predictions = model.predict(X_test)

    # Evaluate Model
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"✅ Model Trained Successfully!")
    print(f"📊 RMSE: {rmse:.2f}")
    print(f"📊 R² Score: {r2:.2f}")

    # Save the trained model
    model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Model saved at: {model_path}")


if __name__ == "__main__":
    ticker = "AAPL"
    df = load_processed_data(ticker)
    train_model(df)
