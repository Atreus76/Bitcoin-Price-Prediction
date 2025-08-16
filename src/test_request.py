import requests

url = "http://127.0.0.1:8000/predict"

payload = {
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

response = requests.post(url, json=payload)
print("Prediction:", response.json())
