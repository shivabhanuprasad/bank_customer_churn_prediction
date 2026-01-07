import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

#🔹 Page Configuration
st.title("🏦 Bank Customer Churn Prediction")
st.write("Predict whether a bank customer is likely to churn.")

#🔹 Load Trained Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")

model = joblib.load(model_path)

#🔹 User Input Section
st.subheader("Customer Information")

credit_score = st.slider("Credit Score", 300, 900, 650)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 80, 35)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Account Balance", min_value=0.0, step=1000.0)
num_products = st.slider("Number of Products", 1, 4, 1)
has_card = st.radio("Has Credit Card?", [0, 1])
is_active = st.radio("Is Active Member?", [0, 1])
salary = st.number_input("Estimated Salary", min_value=0.0, step=1000.0)

#🔹 Convert Input to DataFrame
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_card],
    "IsActiveMember": [is_active],
    "EstimatedSalary": [salary]
})

#🔹 Prediction Button
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer is likely to STAY (Probability: {1 - probability:.2f})")
