from fastapi import FastAPI
import joblib
import pandas as pd

# Initialize app
app = FastAPI(title="Bitcoin Price Predictor", version="1.0")

# Load trained model
model = joblib.load("../models/baseline_lr.pkl")

@app.get("/")
def root():
    return {"message": "Bitcoin Price Prediction API is running 🚀"}

@app.post("/predict")
def predict(data: dict):
    """
    Example input:
    {
      "features": {
        "open": 50000,
        "high": 50500,
        "low": 49500,
        "volume": 12345,
        "hour": 14,
        "day_of_week": 2,
        "is_weekend": 0
      }
    }
    """
    features = pd.DataFrame([data["features"]])
    prediction = model.predict(features)[0]
    return {"prediction": float(prediction)}
