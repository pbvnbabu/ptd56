import streamlit as st
import pandas as pd
import torch
import os
from model import StudentNet
from predict import predict

# Load trained model
model = StudentNet()
model.load_state_dict(torch.load("student_model.pth"))
model.eval()

st.title("🎓 Student Performance Predictor with Admin Dashboard")

# --- Single Prediction ---
st.header("🔍 Predict Single Student")
hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=5.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0)
participation = st.number_input("Participation (0–10)", min_value=0.0, max_value=10.0, value=5.0)
previous_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0, value=60.0)

if st.button("Predict"):
    features = [hours_studied, attendance, participation, previous_score]
    label, confidence = predict(model, features)
    st.success(f"🎯 Prediction: **{label}** with confidence **{confidence * 100:.2f}%**")

# --- Batch Prediction ---
st.header("📂 Predict from CSV")
uploaded_file = st.file_uploader("Upload CSV (columns: hours_studied, attendance, participation, previous_score)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    results = []
    for _, row in df.iterrows():
        features = [
            row["hours_studied"],
            row["attendance"],
            row["participation"],
            row["previous_score"]
        ]
        label, confidence = predict(model, features)
        results.append({
            **row.to_dict(),
            "prediction": label,
            "confidence": f"{confidence * 100:.2f}%"
        })

    result_df = pd.DataFrame(results)
    st.subheader("📊 Batch Predictions")
    st.dataframe(result_df)

    csv_out = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results", csv_out, "predictions.csv", "text/csv")

# --- Admin Dashboard ---
st.header("📜 Prediction Log Viewer")

if os.path.exists("prediction_log.csv"):
    logs = pd.read_csv("prediction_log.csv")

    st.metric("Total Predictions", len(logs))
    st.metric("Fails", sum(logs["prediction"] == "FAIL"))
    st.metric("Avg Confidence", f"{logs['confidence'].mean():.2f}%")

    st.dataframe(logs)

    if st.checkbox("Show only FAIL predictions"):
        st.dataframe(logs[logs["prediction"] == "FAIL"])

    st.download_button("📥 Download Log CSV", logs.to_csv(index=False), "prediction_log.csv", "text/csv")
else:
    st.info("No predictions logged yet.")