import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path

# 1. Load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 2. Create windowed dataset
def create_windowed_data(series, window_size=10):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

# 3. Save baseline results
def save_results(results):
    rmse, mae, mape = results
    with open('results.txt', '+a') as file:
        file.write(f'Baseline Model: Linear Regression\n')
        file.write(f'RMSE: {rmse}\n')
        file.write(f'MAE: {mae}\n')
        file.write(f'MAPE: {mape}')
    return True
# 4. Train baseline model
def train_baseline(data_path, model_path="../models/baseline_lr.pkl", window_size=10):
    df = load_data(data_path)
    close_prices = df['close'].values

    X, y = create_windowed_data(close_prices, window_size)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    results = (rmse, mae, mape)
    save_results(results)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Path("../models").mkdir(exist_ok=True)
    # joblib.dump(model, model_path)

    return model, (rmse, mae, mape)

if __name__ == "__main__":
    train_baseline("data/clean_btc_data.csv", window_size=10)
