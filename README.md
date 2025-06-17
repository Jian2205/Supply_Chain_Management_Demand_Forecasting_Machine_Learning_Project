# Supply Chain Management Demand Forecasting - Machine Learning Project

**Author:** Jianlumei Kamei  
**Internship Organization:** Unified Mentor  
**Duration:** 1 Month  
**Project:** Forecasting product demand using machine learning

---

## Overview

This project aims to build a machine learning model that can predict the number of products sold in a supply chain. Accurate demand forecasting helps improve inventory management, reduce overstocking/understocking, and optimize supply chain operations.

---

## Tools & Technologies Used

- **Language**: Python  
- **IDE**: Jupyter Notebook  
- **Libraries**:
  - `pandas`, `numpy` – Data preprocessing and analysis
  - `matplotlib`, `seaborn` – Data visualization
  - `scikit-learn` – Machine learning models and evaluation
  - `joblib` – Model serialization
  - `streamlit` – Dashboard deployment for user interaction

---

## Project Workflow

1. **Data Cleaning**  
   - Removed missing and redundant data  
   - Encoded categorical variables  
   - Handled outliers

2. **Exploratory Data Analysis (EDA)**  
   - Distribution plots  
   - Correlation analysis  
   - Boxplots and histograms

3. **Feature Engineering**  
   - One-hot encoding  
   - Feature importance analysis  
   - Feature selection based on correlation and model importance

4. **Model Building**  
   - Linear Regression  
   - Random Forest Regressor  
   - Hyperparameter tuning using `RandomizedSearchCV`

5. **Model Evaluation**  
   - RMSE (Root Mean Squared Error)  
   - R² Score  
   - Visual comparisons of different models

6. **Deployment**  
   - Developed a **Streamlit dashboard** to input product data and forecast demand interactively

---

## Model Performance (Tuned Random Forest)

- **RMSE**: 360.85  
- **R² Score**: -0.36  

---

## Conclusion

In this project, I explored demand forecasting using machine learning techniques on real-world supply chain data. I built and evaluated several models, including a tuned Random Forest Regressor. Although the final model resulted in an RMSE of **360.85** and an R² score of **-0.36**, this outcome provided key insights into the challenges of modeling demand with the available data.

### Key Insights:

- A negative R² score indicates the model performs worse than a simple average baseline.
- The RMSE shows significant error in predicted values.
- This may be due to data limitations such as:
  - Lack of time-based features or seasonality information
  - Presence of outliers and noisy features
  - Imbalanced or insufficiently diverse training data

### Learning & Impact:

- Learned to implement the **end-to-end ML pipeline**
- Gained hands-on experience in:
  - Data preprocessing
  - Feature engineering
  - Model evaluation
  - Interactive app deployment using Streamlit
- Developed a working baseline model that can be improved further with future iterations

This project laid a solid foundation for building scalable and accurate demand forecasting systems in real-world supply chain environments.

---

## Repository Contents

- **Jupyter Notebooks**: Each step from data cleaning to final evaluation
- **Data Files**: Cleaned dataset, input samples
- **Model Files**: Trained and tuned model (`.pkl`)
- **Plots**: Evaluation and comparison charts
- **Streamlit App**: For interactive forecasting
