from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

print("Loading trained model...")

model = joblib.load("models/isolation_model.pkl")

print("Model loaded successfully")


@app.get("/")
def home():
    return {"message": "AnomalyLens API is running"}


@app.post("/predict")
def predict(amount: float, time: float):

    data = pd.DataFrame({
        "Time": [time],
        "Amount": [amount]
    })

    # Create dummy V1–V28 columns
    for i in range(1, 29):
        data[f"V{i}"] = 0

    data = data[
        ["Time"] +
        [f"V{i}" for i in range(1, 29)] +
        ["Amount"]
    ]

    prediction = model.predict(data)

    if prediction[0] == -1:
        return {"result": "Anomaly detected"}
    else:
        return {"result": "Normal transaction"}