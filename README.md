# 🔍 AnomalyLens — Intelligent Anomaly Detection System

AnomalyLens is a machine learning-powered web application designed to detect unusual patterns in transactional data using advanced unsupervised learning algorithms.
The system provides real-time anomaly detection, performance comparison across models, and interactive visual analytics through a professional dashboard interface.

---

## 🚀 Live Demo

https://anomalylens-5spriwxnfzb35vqdfxtex8.streamlit.app/
---

## 📌 Features

* Multiple anomaly detection algorithms:

  * Isolation Forest
  * Local Outlier Factor (LOF)
  * DBSCAN

* Algorithm comparison dashboard

* Risk percentage calculation

* Processing time and speed metrics

* Interactive visualization

* CSV dataset upload

* Download detected anomalies

* Real-time performance monitoring

---

## 🧠 Machine Learning Models Used

### 1) Isolation Forest

* Tree-based anomaly detection
* Works well for high-dimensional data
* Fast and scalable

### 2) Local Outlier Factor (LOF)

* Density-based anomaly detection
* Detects local deviations in data

### 3) DBSCAN

* Clustering-based anomaly detection
* Identifies noise points as anomalies

---

## 📊 Dataset

The system uses the **Credit Card Fraud Detection Dataset**, which contains anonymized transaction data.

Dataset characteristics:

* 284,807 transactions
* Highly imbalanced data
* Real-world fraud detection scenario

Features include:

* Time
* Transaction Amount
* PCA-transformed variables (V1–V28)

---

## 🏗️ System Architecture

User
↓
Streamlit Dashboard
↓
Machine Learning Models
↓
Performance Metrics
↓
Visualization & Results

---

## 🧰 Tech Stack

### Frontend

* Streamlit

### Backend / Machine Learning

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

### Deployment

* Streamlit Community Cloud
* GitHub

---

## 📁 Project Structure

AnomalyLens/

```
frontend/
    app.py

models/
    isolation_model.pkl

assets/
    logo.png

requirements.txt
README.md
```

---

## ⚙️ Installation (Local Setup)

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AnomalyLens.git
cd AnomalyLens
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run frontend/app.py
```

---

## 📈 Example Output

The system provides:

* Total Records
* Number of Anomalies
* Risk Percentage
* Processing Time
* Processing Speed
* Visualization Graph
* Algorithm Comparison Table

---

## 🎯 Use Cases

* Fraud detection
* Financial transaction monitoring
* Cybersecurity anomaly detection
* Network intrusion detection
* System health monitoring

---

## 🔮 Future Improvements

* Real-time streaming data detection
* Model hyperparameter tuning
* Advanced visualization dashboards
* Model explainability (SHAP / feature importance)
* Cloud-based API integration

---

## 👨‍💻 Author

Kaustubh Choudhary
B.E. Computer Science (Artificial Intelligence & Machine Learning)

---

## 📜 License

This project is intended for educational and demonstration purposes.
