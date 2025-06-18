import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model
with open("tuned_random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the columns used for training
with open("model_features.pkl", "rb") as file:
    model_columns = pickle.load(file)

st.set_page_config(page_title="Demand Forecasting", layout="wide")

st.title("📦 Demand Forecasting App")
st.markdown("Enter the product details below to predict the number of products that will be sold.")

# Upload CSV
uploaded_file = st.file_uploader("Or upload a CSV file with the input format:", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    input_df = pd.DataFrame(columns=model_columns)

    for col in model_columns:
        input_df[col] = [st.number_input(f"{col}", value=0.0)]

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Number of Products Sold: **{int(prediction)}**")
    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")
