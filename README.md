# 🏦 Bank Customer Churn Prediction & Analysis

A comprehensive, end-to-end Machine Learning web application designed to predict bank customer churn, visualize data insights, and provide actionable retention strategies.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E)
![Plotly](https://img.shields.io/badge/Plotly-Data%20Visualization-3F4F75)

## 📖 Overview

Customer churn (when a customer stops using a company's services) is a major issue in the banking sector. This project uses machine learning to identify customers who are likely to churn, allowing the bank to take proactive measures to retain them.

The project features a **Streamlit front-end** for a premium dashboard experience and a **Scikit-Learn back-end** modeling pipeline.

### Key Features
- **Predictive Analytics:** Uses a robust Random Forest model (trained on customer demographics and account information) to output churn probability.
- **Risk Categorization:** Classifies users into HIGH, MEDIUM, or LOW churn risk.
- **Interactive Dashboards:** Utilizes Plotly to render beautiful, interactive donut charts and bar graphs mapping probability distributions and influencing factors.
- **Automated Recommendations:** Suggests personalized action items based on the customer's predicted risk level.
- **Modular Codebase:** Data preprocessing, model training, and evaluation are split across different source files (`src/`) to simulate a real-world analytics project structure.

---

## 🏗️ Project Structure

```text
bank_customer_churn_prediction/
│
├── app.py                     # Main Streamlit web application
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies (if available)
│
├── data/
│   └── raw/                   # Raw customer dataset (bank_customer_churn.csv)
│
├── models/
│   └── churn_model.pkl        # Serialized, trained Random Forest pipeline
│
├── notebooks/
│   └── EDA.ipynb              # Exploratory Data Analysis & experiments
│
└── src/
    ├── evaluate_model.py      # Script for validating model metrics
    ├── feature_engineering.py # Data engineering placeholder
    ├── preprocessing.py       # Data transformation and scaling pipeline
    └── train_model.py         # Model training script
```

---

## 🚀 Quick Start

### 1. Requirements

Ensure you have Python 3.8+ installed. Install the necessary packages via terminal:

```bash
pip install -r requirements.txt
```
*(Alternatively, you can manually install the required libraries: `pip install streamlit pandas scikit-learn joblib plotly seaborn matplotlib`)*

### 2. Run the Application

Navigate to the root directory folder and launch the Streamlit app:

```bash
streamlit run app.py
```

### 3. Usage & Authentication

Upon visiting the local host URL provided by Streamlit, you will be met with an employee-restricted login screen.

**Demo Credentials**:
- **Username:** `bankemp`
- **Password:** `churn123`

Once logged in, you can adjust customer profile inputs via the sidebar to see real-time updates to their churn risk and corresponding recommendations.

---

## 🧠 Model Pipeline Details

The underlying model is trained using **Scikit-Learn**. 

1. **Preprocessing (`src/preprocessing.py`)**: Applies robust imputation. Continuous numeric values are scaled via `StandardScaler`, and categorical string values are transformed into numerics using `OneHotEncoder`. This is bundled safely into an sklearn `ColumnTransformer`.
2. **Training (`src/train_model.py`)**: Compares a `LogisticRegression` against a `RandomForestClassifier`. The Random Forest estimator operates with `class_weight="balanced"` to prevent bias against the minority class (churners) and encapsulates preprocessing logic within a master scikit-learn `Pipeline`. Data leakage is completely mitigated.
3. **Evaluation (`src/evaluate_model.py`)**: Metrics outputs include Confusion Matrices, ROC-AUC curves, and standard Precision/Recall reports to accurately validate the classifier's capabilities.

---

## ✨ Future Improvements

- Add Database integration instead of hardcoded validation lists for employee authentication.
- Experiment with XGBoost or LightGBM for potentially faster / more accurate multi-variate correlations.
- Expand `feature_engineering.py` logic to create cross-interactions (e.g., balance per total product utilized).
