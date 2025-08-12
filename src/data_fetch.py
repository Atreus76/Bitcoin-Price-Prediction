import os
import pandas as pd
import sqlite3
from binance.client import Client
from datetime import datetime
from typing import Literal

# ===== Binance API Client =====
# Public data only -> keys not required
client = Client(api_key=None, api_secret=None)

# ===== Config =====
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "btc_data.db")

# ===== Functions =====
def get_historical_data(
    symbol: str = "BTCUSDT",
    interval: Literal["1m", "1h", "1d"] = "1d",
    limit: int = 1000,
    save_csv: bool = True,
    save_sqlite: bool = True
) -> pd.DataFrame:
    """
    Download historical OHLCV data from Binance.

    Args:
        symbol: Trading pair, default BTCUSDT.
        interval: Candle interval (1m, 1h, 1d).
        limit: Number of data points (max 1000 per request).
        save_csv: Save to CSV file.
        save_sqlite: Save to SQLite DB.

    Returns:
        DataFrame with historical data.
    """
    # Map to Binance interval constants
    interval_map = {
        "1m": Client.KLINE_INTERVAL_1MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY
    }
    if interval not in interval_map:
        raise ValueError("Interval must be one of '1m', '1h', '1d'")

    print(f"Fetching historical data: {symbol}, {interval}, limit={limit}")
    klines = client.get_klines(symbol=symbol, interval=interval_map[interval], limit=limit)

    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    # Format data
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    # Save to CSV
    if save_csv:
        csv_path = os.path.join(DATA_DIR, f"{symbol}_{interval}_historical.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

    # Save to SQLite
    if save_sqlite:
        conn = sqlite3.connect(DB_PATH)
        table_name = f"{symbol.lower()}_{interval}"
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        print(f"Saved to SQLite table: {table_name}")

    return df


def get_latest_price(symbol: str = "BTCUSDT") -> float:
    """
    Get the latest price for the given symbol.

    Args:
        symbol: Trading pair, default BTCUSDT.

    Returns:
        Latest price as float.
    """
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker["price"])


# ===== Script usage example =====
if __name__ == "__main__":
    # Download last 1000 daily candles
    df_hist = get_historical_data(interval="1d", limit=1000)
    print(df_hist.head())

    # Fetch latest price
    latest_price = get_latest_price()
    print(f"Latest BTC/USDT price: {latest_price}")
