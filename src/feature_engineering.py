import pandas as pd
import numpy as np
import datetime as dt
from EDA import visualize_features
from data_fetch import save_to_csv

df = pd.read_csv('data/clean_btc_data.csv', parse_dates=["timestamp"])

# print(df.head())
# 1. Time based features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# print(df['hour'].head())
# print(df['day_of_week'].head())
# print(df['is_weekend'].head())

# 2. Price returns & log returns
# return = (close - prev_close) / prev_close
# log_return = log(close / prev_close) (reduces skew)

df['return'] = df['close'].pct_change()
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# print(df['return'].head())
# print(df['log_return'].head())

# 3. Rolling window statistics
df['rolling_mean_close_5'] = df['close'].rolling(window=5).mean()
df['rolling_std_close_5'] = df['close'].rolling(window=5).std()

# print(df['rolling_mean_close_5'].head())
# print(df['rolling_std_close_5'].head())

# 4.Technical Indicators (from trading)
df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()

delta = pd.to_numeric(df['close'].diff(), errors='coerce')
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# print(df['ema_10'].head())
# print(df['rsi'].head())

# 5. Lag features
df['close_lag_1'] = df['close'].shift(1)
df['close_lag_3'] = df['close'].shift(3)

# print(df['close_lag_1'].head())
# print(df['close_lag_3'].head())

df = df.dropna().reset_index(drop=True)
print(df.head())
save_to_csv(df, 'data/feature_data.csv')
# visualize_features(df)