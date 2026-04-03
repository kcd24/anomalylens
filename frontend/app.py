import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from PIL import Image

logo = Image.open("assets/logo.jpg")

st.sidebar.image(
    logo,
    use_container_width=True
)
st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### System Info

    **Version:** 1.0  
    **Models:**  
    - Isolation Forest  
    - Local Outlier Factor  
    - DBSCAN  

    **Dataset:** Credit Card Transactions  
    """
)
st.set_page_config(page_title="AnomalyLens",page_icon="🔍", layout="wide")

st.markdown(
    """
    # AnomalyLens  
    ### Intelligent Anomaly Detection System

    Detect unusual patterns in financial transactions using advanced machine learning algorithms.
    """
)

# Sidebar settings
st.sidebar.header("Model Settings")
algorithm = st.sidebar.selectbox(
    "Choose Algorithm",
    (
        "Isolation Forest",
        "Local Outlier Factor",
        "DBSCAN"
    )
)
compare_models = st.sidebar.checkbox(
    "Compare All Algorithms"
)

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(data.head())

    if st.button("Run Anomaly Detection"):
        start_time = time.time()
        with st.spinner("Analyzing transactions..."):

            X = data.drop("Class", axis=1)

            # Select algorithm

            if algorithm == "Isolation Forest":

                model = IsolationForest(
                    contamination=0.01,
                    random_state=42
                )

                predictions = model.fit_predict(X)

            elif algorithm == "Local Outlier Factor":

                model = LocalOutlierFactor(
                    n_neighbors=20,
                    contamination=0.01
                )

                predictions = model.fit_predict(X)

            elif algorithm == "DBSCAN":

                model = DBSCAN(
                    eps=10,
                    min_samples=5
                )

                predictions = model.fit_predict(X)

            # Save predictions
            data["anomaly"] = predictions

            # Standardize labels
            data["anomaly"] = data["anomaly"].apply(
                lambda x: -1 if x == -1 else 1
            )

            anomalies = data[data["anomaly"] == -1]
            normal = data[data["anomaly"] == 1]

        st.success("Detection Complete")
        end_time = time.time()

        processing_time = end_time - start_time

        records_per_second = len(data) / processing_time

        # Metrics

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Records",
                len(data)
            )

        with col2:
            st.metric(
                "Anomalies Detected",
                len(anomalies)
            )

        risk_percent = (
            len(anomalies) / len(data)
        ) * 100

        with col3:
            st.metric(
                "Risk Percentage",
                f"{risk_percent:.2f}%"
            )

        with col4:
            st.metric(
                "Processing Time",
                f"{processing_time:.2f} sec"
            )
            st.caption(
                f"Processing Speed: {records_per_second:,.0f} records/sec"
            )
        if compare_models:

            results = []

            algorithms = [
                "Isolation Forest",
                "Local Outlier Factor",
                "DBSCAN"
            ]

            for algo in algorithms:

                start = time.time()

                if algo == "Isolation Forest":

                    model = IsolationForest(
                        contamination=0.01,
                        random_state=42
                    )

                    preds = model.fit_predict(X)

                elif algo == "Local Outlier Factor":

                    model = LocalOutlierFactor(
                        n_neighbors=20,
                        contamination=0.01
                    )

                    preds = model.fit_predict(X)

                elif algo == "DBSCAN":

                    model = DBSCAN(
                        eps=10,
                        min_samples=5
                    )

                    preds = model.fit_predict(X)

                preds = [-1 if x == -1 else 1 for x in preds]

                anomalies_count = preds.count(-1)

                end = time.time()

                processing_time = end - start

                risk_percent = (
                    anomalies_count / len(X)
                ) * 100

                results.append({
                    "Algorithm": algo,
                    "Anomalies": anomalies_count,
                    "Risk %": round(risk_percent, 2),
                    "Time (sec)": round(processing_time, 2)
                })

            st.subheader("Algorithm Comparison")

            comparison_df = pd.DataFrame(results)

            st.dataframe(comparison_df)

            st.stop()
        
        # Visualization

        st.subheader("Visualization")

        fig = plt.figure()

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

        plt.xlabel("Transaction Amount")
        plt.ylabel("Time")
        plt.title(
            f"Transaction Anomaly Detection ({algorithm})"
        )

        st.pyplot(fig)

        # Table

        st.subheader("Anomalies Table")

        st.dataframe(anomalies.head(50))

        # Download

        csv = anomalies.to_csv(index=False)

        st.download_button(
            label="Download Anomalies CSV",
            data=csv,
            file_name="anomalies_detected.csv",
            mime="text/csv"
        )
        

st.markdown("---")

st.markdown(
    """
    **AnomalyLens v1.0**  
    Built using Machine Learning and Streamlit  
    """
)