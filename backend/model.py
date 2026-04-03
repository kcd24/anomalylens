import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import joblib

print("Loading dataset...")

data = pd.read_csv("data/creditcard.csv")

print("Dataset loaded successfully")

# Remove label column
X = data.drop("Class", axis=1)

print("Training Isolation Forest model...")

model = IsolationForest(
    contamination=0.01,
    random_state=42
)

model.fit(X)
joblib.dump(model, "models/isolation_model.pkl")
print("Model saved successfully")

print("Detecting anomalies...")

data["anomaly"] = model.predict(X)

print("\nAnomaly Results:")
print(data["anomaly"].value_counts())

# -------------------------
# Visualization
# -------------------------

print("Creating visualization...")

anomalies = data[data["anomaly"] == -1]
normal = data[data["anomaly"] == 1]

plt.figure()

plt.scatter(
    normal["Amount"],
    normal["Time"],
    alpha=0.5
)

plt.scatter(
    anomalies["Amount"],
    anomalies["Time"],
    alpha=0.8
)

plt.title("Transaction Anomaly Detection")

plt.xlabel("Transaction Amount")

plt.ylabel("Time")

plt.show()