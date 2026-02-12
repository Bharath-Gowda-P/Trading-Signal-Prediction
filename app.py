import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and feature names
model = joblib.load("xgb_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="XGBoost Buy Signal Predictor", layout="centered")
st.title("📈 XGBoost Buy / No-Buy Prediction")

st.write("Enter feature values used during training:")

# Collect inputs dynamically
input_data = {}

for feature in feature_columns:
    input_data[feature] = st.number_input(
        label=feature,
        value=0.0
    )

# Convert to DataFrame (VERY IMPORTANT for XGBoost)
input_df = pd.DataFrame([input_data])

# Threshold slider (matches your experiments)
threshold = st.slider(
    "Prediction Threshold",
    min_value=0.50,
    max_value=0.80,
    value=0.60,
    step=0.01
)

if st.button("Predict"):
    # Probability for class 1 (BUY)
    prob = model.predict_proba(input_df)[:, 1][0]

    prediction = int(prob > threshold)

    st.subheader("📊 Result")
    st.write(f"**Buy Probability:** `{prob:.4f}`")
    st.write(f"**Threshold Used:** `{threshold}`")

    if prediction == 1:
        st.success("✅ BUY SIGNAL")
    else:
        st.error("❌ NO BUY SIGNAL")
