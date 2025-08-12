import os
import pandas as pd
from binance.client import Client
from datetime import datetime
from EDA import load_and_clean_data, visualize_features

# Binance API credentials (read from env vars for security)
# BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
# BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

client = Client(api_key=None, api_secret=None)

def fetch_historical_data(symbol="BTCUSDT", interval='1h', start_str="1 Jan 2022"):
    """
    Fetch historical BTC price data from Binance API.
    Returns a pandas DataFrame.
    """
    klines = client.get_historical_klines(symbol, interval, start_str)

    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    # Convert timestamp to readable date
    # df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    # df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    # Convert numeric columns to float
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    df["num_trades"] = df["num_trades"].astype(int)

    return df

def save_to_csv(df, path):
    """Save DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

def main():
    raw_csv_path = "data/raw_btc_data.csv"
    clean_csv_path = "data/clean_btc_data.csv"

    os.makedirs("data", exist_ok=True)

    print("Fetching historical BTC data...")
    df_raw = fetch_historical_data()
    save_to_csv(df_raw, raw_csv_path)

    print("Cleaning data...")
    df_clean = load_and_clean_data(raw_csv_path)
    save_to_csv(df_clean, clean_csv_path)

    print("Visualizing important features...")
    visualize_features(df_clean)

if __name__ == "__main__":
    main()
