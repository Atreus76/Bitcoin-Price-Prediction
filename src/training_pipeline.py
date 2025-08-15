# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# import xgboost as xgb

# # ---------- 1. Metrics ----------
# def evaluate_model(y_true, y_pred):
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = mean_absolute_error(y_true, y_pred)
#     mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#     return rmse, mae, mape

# # ---------- 2. Pipeline ----------
# def train_pipeline(df, model):
#     # 1. Define target
#     df = df.copy()
#     df['target'] = df['close'].shift(-1)
#     df = df.dropna()

#     # 2. Features (drop timestamp & target)
#     feature_cols = [col for col in df.columns if col not in ['timestamp', 'target', 'close_time']]
#     X = df[feature_cols]
#     y = df['target']

#     # 3. Train-test split (by time)
#     train_size = int(len(df) * 0.8)
#     X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
#     y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

#     # 4. Fit model
#     model.fit(X_train, y_train)

#     # 5. Predict
#     y_pred = model.predict(X_test)

#     # 6. Evaluate
#     rmse, mae, mape = evaluate_model(y_test, y_pred)
#     print(f"RMSE: {rmse:.4f}")
#     print(f"MAE: {mae:.4f}")
#     print(f"MAPE: {mape:.2f}%")

#     return model, (rmse, mae, mape)

# # ---------- 3. Run ----------
# if __name__ == "__main__":
#     df = pd.read_csv("data/feature_data.csv", parse_dates=['timestamp'])

#     print("\n--- Linear Regression ---")
#     train_pipeline(df, LinearRegression())

#     print("\n--- Random Forest ---")
#     train_pipeline(df, RandomForestRegressor(n_estimators=100, random_state=42))

#     print("\n--- XGBoost ---")
#     train_pipeline(df, xgb.XGBRegressor(n_estimators=100, random_state=42))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def train_models(df, target_col='close'):
    
    # 1. Define target
    df = df.copy()
    df['target'] = df['close'].shift(-1)
    df = df.dropna()

    # 2. Features (drop timestamp & target)
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'target', 'close_time']]
    X = df[feature_cols]
    y = df['target']

    # Time-based split (80% train, 20% test)
    split_index = int(len(df) * 0.8)
    train, test = df.iloc[:split_index], df.iloc[split_index:]

    X_train, y_train = train[feature_cols], train['target']
    X_test, y_test = test[feature_cols], test['target']
    print(X_test.head())
    print(y_test.head())
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Check the results of naive model
    preds = X_test['close'].shift(-1).fillna(y_test.iloc[0])
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    
    print(f"\n--- Naive ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    def evaluate_model(model, name):
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        print(f"\n--- {name} ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")

        return model, preds

    # Linear Regression
    lr = LinearRegression()
    lr_model, _ = evaluate_model(lr, "Linear Regression")

    # Random Forest (tuned)
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model, _ = evaluate_model(rf, "Random Forest")

    # XGBoost (tuned)
    xgb = XGBRegressor(
        n_estimators=5000,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model, _ = evaluate_model(xgb, "XGBoost")

    # Feature Importance (Tree models)
    for name, model in [("Random Forest", rf_model), ("XGBoost", xgb_model)]:
        importance = model.feature_importances_
        features = X_train.columns
        importance_df = pd.DataFrame({"Feature": features, "Importance": importance})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(8, 5))
        plt.barh(importance_df["Feature"], importance_df["Importance"])
        plt.gca().invert_yaxis()
        plt.title(f"{name} Feature Importance")
        # plt.show()

    return lr_model, rf_model, xgb_model

df = pd.read_csv("data/feature_data.csv", parse_dates=['timestamp'])
train_models(df)
