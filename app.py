{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 
   "id": "322068c5-72a2-4b7b-8c75-4ab31010921e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "# Load the trained model and model features\n",
    "with open(\"tuned_random_forest_model.pkl\", \"rb\") as file:\n",
    "    model = pickle.load(file)\n",
    "\n",
    "with open(\"model_features.pkl\", \"rb\") as f:\n",
    "    model_features = pickle.load(f)\n",
    "\n",
    "# App title\n",
    "st.set_page_config(page_title=\"Supply Chain Demand Forecasting\", layout=\"wide\")\n",
    "st.title(\"📦 Supply Chain Demand Forecasting\")\n",
    "st.markdown(\"This dashboard uses a Tuned Random Forest Regressor to forecast product demand.\")\n",
    "\n",
    "# Sample CSV format help\n",
    "with st.expander(\"📄 Click to see sample input format\"):\n",
    "    st.markdown(\"\"\"\n",
    "    Example input CSV file must include these columns:  \n",
    "    - Product Type  \n",
    "    - Warehouse  \n",
    "    - Mode of Shipment  \n",
    "    - Customer Rating  \n",
    "    - Discount Offered  \n",
    "    - Weight in gms  \n",
    "    - Prior Purchases  \n",
    "    - Product Importance  \n",
    "    - Gender  \n",
    "    - Customer Care Calls  \n",
    "    - etc. (as per your encoded model features)\n",
    "    \"\"\")\n",
    "\n",
    "# Upload CSV file\n",
    "uploaded_file = st.file_uploader(\"Upload input CSV file to predict demand\", type=[\"csv\"])\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    input_df = pd.read_csv(uploaded_file)\n",
    "\n",
    "    # Ensure columns match the training features\n",
    "    input_df = input_df.reindex(columns=model_features, fill_value=0)\n",
    "\n",
    "    # Make prediction\n",
    "    predictions = model.predict(input_df)\n",
    "\n",
    "    # Show output\n",
    "    input_df['Predicted Demand'] = np.round(predictions, 2)\n",
    "    st.success(\"✅ Prediction completed!\")\n",
    "    st.dataframe(input_df)\n",
    "\n",
    "    # Download result\n",
    "    csv = input_df.to_csv(index=False).encode('utf-8')\n",
    "    st.download_button(\"📥 Download Predictions as CSV\", data=csv, file_name=\"predicted_demand.csv\", mime=\"text/csv\")\n",
    "else:\n",
    "    st.info(\"👈 Upload a CSV file to get started.\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
