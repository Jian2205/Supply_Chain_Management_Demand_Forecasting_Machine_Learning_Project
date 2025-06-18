import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained tuned model
model = joblib.load("tuned_random_forest_model.pkl")

# Define the exact input features used during training
model_columns = [
    'Price',
    'Stock levels',
    'Order quantities',
    'Shipping costs',
    'Production volumes',
    'Manufacturing costs',
    'Costs',
    'Inspection results_Pass',
    'Promotion_FlashSale',
    'Product type_skincare'
]

st.set_page_config(page_title="📦 Demand Forecasting App", layout="wide")
st.title("📦 Demand Forecasting App")
st.markdown("Enter the product details below to predict the number of products that will be sold.")

# Upload CSV option
uploaded_file = st.file_uploader("📁 Upload a CSV file with required features", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        # Check and align columns
        missing_cols = [col for col in model_columns if col not in input_df.columns]
        extra_cols = [col for col in input_df.columns if col not in model_columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()
        if extra_cols:
            st.warning(f"Ignoring extra columns: {extra_cols}")
            input_df = input_df[model_columns]

        input_df = input_df[model_columns]  # Ensure correct column order

    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        st.stop()

else:
    # Manual input through sidebar
    st.sidebar.header("📋 Manual Input Features")
    input_data = {}
    for col in model_columns:
        input_data[col] = st.sidebar.number_input(f"{col}", value=0.0)

    input_df = pd.DataFrame([input_data])

# Predict and show result
if st.button("🔮 Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted Number of Products Sold: **{int(prediction)}**")
        st.write("📊 Input Data Preview:")
        st.dataframe(input_df)
    except Exception as e:
        st.error(f"❌ Prediction failed. Error: {e}")
