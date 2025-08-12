# eda.py
"""
EDA and data cleaning utilities for BTC price prediction.
This module can be imported in training and prediction scripts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def load_and_clean_data(csv_path: str,
                        use_features: Optional[list[str]] = None,
                        dropna: bool = True) -> pd.DataFrame:
    """
    Load BTC historical price data, clean it, and return a processed DataFrame.
    """
    # Load
    df = pd.read_csv(csv_path)

    # Convert timestamp columns to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    # Sort by time
    df = df.sort_values(by='timestamp')

    # Drop unnecessary columns
    if 'ignore' in df.columns:
        df = df.drop(columns=['ignore'])

    # Handle missing values
    if dropna:
        df = df.dropna()

    # Ensure numeric columns are floats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Feature selection (if specified)
    if use_features is not None:
        df = df[['timestamp'] + [col for col in use_features if col in df.columns]]

    df = df.reset_index(drop=True)
    return df


def describe_data(df: pd.DataFrame):
    """
    Print basic statistics and info about the dataset.
    """
    print("\n===== Data Overview =====")
    print(df.head())
    print("\n===== Data Info =====")
    print(df.info())
    print("\n===== Missing Values =====")
    print(df.isna().sum())
    print("\n===== Statistics =====")
    print(df.describe())


def visualize_features(df: pd.DataFrame):
    """
    Generate key visualizations for BTC data.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(15, 10))

    # Plot Closing Price over Time
    plt.subplot(3, 1, 1)
    plt.plot(df['timestamp'], df['close'], label='Close Price', color='blue')
    plt.title("BTC Closing Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price (USDT)")
    plt.legend()

    # Correlation Heatmap
    plt.subplot(3, 1, 2)
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")

    # Volume vs Price Scatter
    plt.subplot(3, 1, 3)
    plt.scatter(df['volume'], df['close'], alpha=0.3, color='green')
    plt.title("Volume vs Closing Price")
    plt.xlabel("Volume")
    plt.ylabel("Close Price")

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    data = load_and_clean_data("data/btc_usdt_raw.csv")
    describe_data(data)
    visualize_features(data)
