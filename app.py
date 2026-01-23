import streamlit as st
import joblib
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------------------------
# Industry Level UI + Background + Cursor + Theme
# --------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

/* Global */
* {
    font-family: 'Inter', sans-serif;
    cursor: default;
}

button, input, select, textarea {
    cursor: pointer !important;
}

/* Background image */
.stApp {
    background: linear-gradient(rgba(2,6,23,0.85), rgba(2,6,23,0.85)),
                url("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1600&q=80");
    background-size: cover;
    background-attachment: fixed;
}

.block-container {
    padding-top: 2rem;
}

/* Hero */
.hero-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #22c55e);
    -webkit-background-clip: text;
    color: transparent;
}

.hero-subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 40px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(18px);
    border-radius: 22px;
    padding: 28px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 15px 40px rgba(0,0,0,0.6);
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 0 35px rgba(56,189,248,0.5);
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 12px;
    color: #e5e7eb;
}

.kpi {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    border-right: 1px solid rgba(255,255,255,0.1);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* Button */
.stButton > button {
    width: 100%;
    padding: 14px;
    font-size: 17px;
    font-weight: 700;
    border-radius: 16px;
    background: linear-gradient(135deg, #38bdf8, #6366f1);
    color: white;
    border: none;
    box-shadow: 0 0 25px rgba(99,102,241,0.6);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 45px rgba(56,189,248,0.9);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Hero Section
# --------------------------------------------------
st.markdown("<div class='hero-title'>Bank Customer Churn Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Predict • Analyze • Retain customers with AI-powered insights</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models", "churn_model.pkl"))

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("📊 Customer Profile")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 80, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
balance = st.sidebar.number_input("Account Balance", min_value=0.0, step=1000.0)
num_products = st.sidebar.slider("Number of Products", 1, 4, 1)
has_card = st.sidebar.radio("Credit Card", ["Yes", "No"])
is_active = st.sidebar.radio("Active Member", ["Yes", "No"])
salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, step=1000.0)

# --------------------------------------------------
# Input DataFrame
# --------------------------------------------------
input_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [1 if has_card == "Yes" else 0],
    "IsActiveMember": [1 if is_active == "Yes" else 0],
    "EstimatedSalary": [salary]
})

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🚀 Predict Churn Risk"):

    probability = model.predict_proba(input_df)[0][1]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="section-title">Churn Probability</div>
            <div class="kpi">{probability:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        risk = "HIGH" if probability > 0.6 else "MEDIUM" if probability > 0.4 else "LOW"
        color = "#ef4444" if risk == "HIGH" else "#f59e0b" if risk == "MEDIUM" else "#22c55e"

        st.markdown(f"""
        <div class="card">
            <div class="section-title">Risk Level</div>
            <div class="kpi" style="color:{color}">{risk}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        action = "Immediate Retention Strategy" if risk == "HIGH" else "Monitor & Upsell"

        st.markdown(f"""
        <div class="card">
            <div class="section-title">Recommended Action</div>
            <div class="kpi" style="font-size:24px">{action}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="section-title">Prediction Breakdown</div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------
    # Plotly Visualization (Industry grade)
    # --------------------------------------------------

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=["Customer Stays", "Customer Churns"],
        x=[1 - probability, probability],
        orientation='h',
        marker=dict(color=["#22c55e", "#ef4444"]),
        text=[f"{(1-probability)*100:.1f}%", f"{probability*100:.1f}%"],
        textposition='auto'
    ))

    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        xaxis=dict(range=[0,1], title="Probability"),
        margin=dict(l=40, r=40, t=20, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)
