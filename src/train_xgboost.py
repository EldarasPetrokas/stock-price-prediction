import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Get absolute paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


def load_lstm_enhanced_data(ticker):
    """ Load dataset with LSTM features """
    file_path = os.path.join(DATA_DIR, f"{ticker}_with_lstm_features.csv")
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Drop rows with NaN (since LSTM features start late)
    df.dropna(inplace=True)

    # Dynamically select all LSTM features
    lstm_features = [col for col in df.columns if "LSTM_Feature" in col]

    # Define input features & target variable
    features = ["SMA_50", "SMA_200", "EMA_12", "EMA_26", "MACD", "RSI", "Upper_Band", "Lower_Band",
                "ATR"] + lstm_features
    target = "Close"

    X = df[features]
    y = df[target].shift(-1)  # Predict next day's closing price

    # Drop last row (NaN target after shifting)
    X, y = X.iloc[:-1], y.iloc[:-1]

    return X, y


def train_xgboost_model(ticker):
    """ Train an XGBoost Model """
    X, y = load_lstm_enhanced_data(ticker)

    # Split into training & test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
    }

    # Set up GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(xgb.XGBRegressor(objective="reg:squarederror"),
                               param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
    grid_search.fit(X_train, y_train)

    # Best parameters from grid search
    best_params = grid_search.best_params_
    print(f"Best Hyperparameters: {best_params}")

    # Train the model with best parameters
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    # Make Predictions
    predictions = model.predict(X_test)

    # Evaluate Model on the Test Set
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"✅ XGBoost Model Trained!")
    print(f"📊 RMSE: {rmse:.2f}")
    print(f"📊 R² Score: {r2:.2f}")

    # Cross-validation performance
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"📊 Cross-Validation RMSE: {-np.mean(cv_scores):.2f}")

    # Plot Actual vs Predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label="Actual Prices", color='blue')
    plt.plot(y_test.index, predictions, label="Predicted Prices", color='red', linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"XGBoost: Actual vs Predicted Prices for {ticker}")
    plt.legend()
    plt.show()

    # Save the trained model
    model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    model.save_model(model_path)
    print(f"✅ XGBoost Model saved at: {model_path}")


if __name__ == "__main__":
    train_xgboost_model("AAPL")
