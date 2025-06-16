import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Demand Forecasting", layout="wide")

# Load model and feature columns
model = joblib.load('tuned_random_forest_model.pkl')
trained_features = joblib.load('model_features.pkl')

# Sidebar navigation
st.sidebar.title("🚀 Demand Forecasting Dashboard")
page = st.sidebar.radio("Navigation", [
    "Overview",
    "Upload & Preview",
    "Model Performance",
    "Feature Importance",
    "Predict"
])

# Overview page
if page == "Overview":
    st.title("🧠 Demand Forecasting Dashboard")
    st.markdown("""
    This dashboard predicts **Number of Products Sold** using a trained Random Forest model.

    **Objective:** Forecast product demand for better supply chain management.  
    **Model Used:** Tuned Random Forest Regressor  
    **Files Included:** app.py, model pickle, feature list, evaluation plots.
    """)

# Upload & Preview page
elif page == "Upload & Preview":
    st.header("📤 Upload Your CSV Data")
    uploaded_file = st.file_uploader("Upload a CSV file with feature data (no target column)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df.head())

# Model Performance page
elif page == "Model Performance":
    st.header("📊 Model Performance Plots")
    st.image("rmse_comparison_plot.png", caption="RMSE Comparison: Before vs. After Tuning", use_column_width=True)
    st.image("r2_comparison_plot.png", caption="R² Score Comparison: Before vs. After Tuning", use_column_width=True)

# Feature Importance page
elif page == "Feature Importance":
    st.header("📈 Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': trained_features, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feat_df['Feature'], feat_df['Importance'], color='skyblue')
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# Predict page
elif page == "Predict":
    st.header("🔮 Predict New Data")
    input_file = st.file_uploader("Upload new data", type="csv")
    if input_file is not None:
        data = pd.read_csv(input_file)
        if 'Number of products sold' in data.columns:
            data = data.drop(columns=['Number of products sold'])
        data_encoded = pd.get_dummies(data).reindex(columns=trained_features, fill_value=0)
        if st.button("🔮 Predict"):
            try:
                predictions = model.predict(data_encoded)
                data['Predicted Sales'] = predictions
                st.success("✅ Predictions generated successfully!")
                st.subheader("📈 Predicted Results")
                st.dataframe(data[['Predicted Sales']].head())
                st.download_button(
                    label="📥 Download Predictions",
                    data=data.to_csv(index=False),
                    file_name="predicted_sales.csv",
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

st.markdown("---")
st.markdown("🔧 Built by Jianlumei Kamei | Powered by Streamlit")
