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

def visualize_features(df, target_col='close'):
    """Visualize key engineered features for time series data (auto-detect available columns)."""

    # Helper: safe plotting
    def safe_plot(condition, plot_func):
        try:
            if condition:
                plot_func()
        except Exception as e:
            print(f"Skipping plot due to error: {e}")

    # 1. Distribution of returns
    safe_plot('return' in df.columns, lambda: (
        plt.figure(figsize=(8,4)),
        sns.histplot(df['return'].dropna(), bins=100, kde=True),
        plt.title('Distribution of Returns'),
        plt.xlabel('Return'),
        plt.ylabel('Frequency'),
        plt.show()
    ))

    # 2. Distribution of log returns
    safe_plot('log_return' in df.columns, lambda: (
        plt.figure(figsize=(8,4)),
        sns.histplot(df['log_return'].dropna(), bins=100, kde=True),
        plt.title('Distribution of Log Returns'),
        plt.xlabel('Log Return'),
        plt.ylabel('Frequency'),
        plt.show()
    ))

    # 3. Volatility by hour
    safe_plot(all(col in df.columns for col in ['hour', 'return']), lambda: (
        plt.figure(figsize=(8,4)),
        sns.boxplot(x='hour', y='return', data=df),
        plt.title('Return Volatility by Hour'),
        plt.show()
    ))

    # 4. Volatility by day of week
    safe_plot(all(col in df.columns for col in ['day_of_week', 'return']), lambda: (
        plt.figure(figsize=(8,4)),
        sns.boxplot(x='day_of_week', y='return', data=df),
        plt.title('Return Volatility by Day of Week'),
        plt.show()
    ))

    # 5. Correlation heatmap
    safe_plot(target_col in df.columns, lambda: (
        plt.figure(figsize=(10,6)),
        sns.heatmap(
            df.corr(numeric_only=True)[[target_col]].sort_values(by=target_col, ascending=False),
            annot=True, cmap='coolwarm', fmt=".2f"
        ),
        plt.title(f'Correlation with {target_col}'),
        plt.show()
    ))


# Example usage
if __name__ == "__main__":
    data = load_and_clean_data("data/btc_usdt_raw.csv")
    describe_data(data)
    visualize_features(data)
