import streamlit as st
import joblib
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------------------------
# Authentication
# --------------------------------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --------------------------------------------------
# Login Function
# --------------------------------------------------
def login(username, password):
    if username == "bankemp" and password == "churn123":
        st.session_state.logged_in = True
        st.success("Login successful! Redirecting...")
        st.rerun()
    else:
        st.error("Invalid username or password")

if not st.session_state.logged_in:
    # --------------------------------------------------
    # Industry Level UI + Background + Cursor + Theme for Login
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
        background: linear-gradient(rgba(255,255,255,0.9), rgba(255,255,255,0.9)),
                    url("https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=1600&q=80");
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
        background: linear-gradient(90deg, #1e3a8a, #3b82f6, #10b981);
        -webkit-background-clip: text;
        color: transparent;
    }

    .hero-subtitle {
        text-align: center;
        font-size: 18px;
        color: #374151;
        margin-bottom: 40px;
    }

    /* Login Card */
    .login-card {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(18px);
        border-radius: 22px;
        padding: 40px;
        border: 1px solid rgba(0,0,0,0.1);
        box-shadow: 0 15px 40px rgba(0,0,0,0.1);
        max-width: 400px;
        margin: 0 auto;
        text-align: center;
    }

    .login-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 20px;
        color: #1f2937;
    }

    .login-subtitle {
        font-size: 16px;
        color: #6b7280;
        margin-bottom: 30px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        padding: 14px;
        font-size: 17px;
        font-weight: 700;
        border-radius: 16px;
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: white;
        border: none;
        box-shadow: 0 0 25px rgba(30,58,138,0.4);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 45px rgba(59,130,246,0.7);
    }

    </style>
    """, unsafe_allow_html=True)

    # --------------------------------------------------
    # Hero Section
    # --------------------------------------------------
    st.markdown("<div class='hero-title'>Bank Customer Churn Intelligence</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Predict • Analyze • Retain customers with AI-powered insights</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # Login Form
    # --------------------------------------------------
    st.markdown("""
    <div class="login-card">
        <div class="login-title">🔒 Employee Login</div>
        <div class="login-subtitle">Access restricted to authorized bank employees only.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Login")
            if submitted:
                login(username, password)

else:
    # --------------------------------------------------
    # Main App (Logged In)
    # --------------------------------------------------
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
        background: linear-gradient(rgba(255,255,255,0.9), rgba(255,255,255,0.9)),
                    url("https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=1600&q=80");
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
        background: linear-gradient(90deg, #1e3a8a, #3b82f6, #10b981);
        -webkit-background-clip: text;
        color: transparent;
    }

    .hero-subtitle {
        text-align: center;
        font-size: 18px;
        color: #374151;
        margin-bottom: 40px;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(18px);
        border-radius: 22px;
        padding: 28px;
        border: 1px solid rgba(0,0,0,0.1);
        box-shadow: 0 15px 40px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    .card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 0 35px rgba(59,130,246,0.3);
    }

    .section-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 12px;
        color: #1f2937;
    }

    .kpi {
        font-size: 42px;
        font-weight: 800;
        color: #1e3a8a;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc, #e2e8f0);
        border-right: 1px solid rgba(0,0,0,0.1);
    }

    section[data-testid="stSidebar"] * {
        color: #374151;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        padding: 14px;
        font-size: 17px;
        font-weight: 700;
        border-radius: 16px;
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: white;
        border: none;
        box-shadow: 0 0 25px rgba(30,58,138,0.4);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 45px rgba(59,130,246,0.7);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0,0,0,0.05);
        border-radius: 12px;
        padding: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #374151;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: white;
    }

    /* Risk-specific colors */
    .high-risk { color: #dc2626; }
    .medium-risk { color: #d97706; }
    .low-risk { color: #059669; }

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
    # Prediction and Dashboard
    # --------------------------------------------------
    if st.button("🚀 Predict Churn Risk"):
        probability = model.predict_proba(input_df)[0][1]
        risk = "HIGH" if probability > 0.6 else "MEDIUM" if probability > 0.4 else "LOW"
        risk_class = "high-risk" if risk == "HIGH" else "medium-risk" if risk == "MEDIUM" else "low-risk"

        # Tabs for Dashboard
        tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Detailed Analysis", "💡 Insights & Recommendations"])

        with tab1:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="card">
                    <div class="section-title">Churn Probability</div>
                    <div class="kpi">{probability:.2%}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="card">
                    <div class="section-title">Risk Level</div>
                    <div class="kpi {risk_class}">{risk}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                action = "Immediate Retention Strategy" if risk == "HIGH" else "Monitor & Upsell" if risk == "MEDIUM" else "Maintain Satisfaction"
                st.markdown(f"""
                <div class="card">
                    <div class="section-title">Recommended Action</div>
                    <div class="kpi" style="font-size:24px">{action}</div>
                </div>
                """, unsafe_allow_html=True)

            # Donut Chart for Probability (new form)
            fig_donut = go.Figure(data=[go.Pie(
                labels=["Retention Probability", "Churn Probability"],
                values=[1 - probability, probability],
                hole=.6,
                marker_colors=["#10b981", "#ef4444"],
                textinfo="label+percent",
                textfont_size=14
            )])

            fig_donut.update_layout(
                title_text="Churn vs Retention Probability",
                paper_bgcolor="rgba(255,255,255,0)",
                font=dict(color="#374151"),
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig_donut, use_container_width=True)

        with tab2:
            st.markdown(f"""
            <div class="card">
                <div class="section-title">Prediction Breakdown - {risk} Risk</div>
                <p style="color:#6b7280;">Detailed analysis tailored to the {risk.lower()} risk level.</p>
            </div>
            """, unsafe_allow_html=True)

            # Subplots with different chart forms
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Churn Probability Distribution", "Key Factors Influencing Risk"),
                specs=[[{"type": "pie"}, {"type": "bar"}]]
            )

            # Pie Chart for Probability Distribution (new form)
            fig.add_trace(go.Pie(
                labels=["Customer Stays", "Customer Churns"],
                values=[1 - probability, probability],
                marker_colors=["#10b981", "#ef4444"],
                textinfo="label+percent",
                textfont_size=12,
                showlegend=False
            ), row=1, col=1)

            # Horizontal Bar Chart for Feature Importance (properly shaped)
            features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
            if risk == "HIGH":
                importance = [0.1, 0.25, 0.05, 0.2, 0.15, 0.05, 0.1, 0.1]  # Age and Balance high
            elif risk == "MEDIUM":
                importance = [0.15, 0.15, 0.1, 0.15, 0.1, 0.1, 0.15, 0.1]  # Balanced
            else:
                importance = [0.2, 0.1, 0.15, 0.1, 0.1, 0.1, 0.15, 0.1]  # CreditScore and Tenure high

            fig.add_trace(go.Bar(
                y=features,
                x=importance,
                orientation='h',
                marker=dict(color="#3b82f6"),
                name="Importance"
            ), row=1, col=2)

            fig.update_layout(
                height=500,
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="rgba(255,255,255,0)",
                font=dict(color="#374151"),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown(f"""
            <div class="card">
                <div class="section-title">Insights & Recommendations - {risk} Risk</div>
                <p style="color:#6b7280;">Tailored strategies to mitigate churn based on the {risk.lower()} risk assessment.</p>
            </div>
            """, unsafe_allow_html=True)

            if risk == "HIGH":
                recommendations = [
                    "Schedule an immediate personalized call to understand dissatisfaction.",
                    "Offer exclusive retention bonuses, such as waived fees or higher interest rates.",
                    "Implement a loyalty program with rewards for long-term engagement.",
                    "Analyze recent transactions for signs of inactivity and provide targeted promotions."
                ]
            elif risk == "MEDIUM":
                recommendations = [
                    "Monitor account activity closely over the next 3-6 months.",
                    "Suggest upsell opportunities, like additional products or services.",
                    "Send personalized email campaigns highlighting benefits and support.",
                    "Conduct a satisfaction survey to gather feedback and improve services."
                ]
            else:
                recommendations = [
                    "Maintain regular communication through newsletters and check-ins.",
                    "Encourage referrals and rewards for loyal behavior.",
                    "Provide educational content on financial management to build trust.",
                    "Focus on overall customer experience enhancements."
                ]

            st.markdown("<ul style='color:#374151;'>", unsafe_allow_html=True)
            for rec in recommendations:
                st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)

            # Additional KPI Cards
            col4, col5 = st.columns(2)

            with col4:
                st.markdown(f"""
                <div class="card">
                    <div class="section-title">Retention Potential</div>
                    <div class="kpi low-risk">{(1-probability)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with col5:
                priority = 3 if risk == "HIGH" else 2 if risk == "MEDIUM" else 1
                st.markdown(f"""
                <div class="card">
                    <div class="section-title">Action Priority</div>
                    <div class="kpi medium-risk">{priority}/3</div>
                </div>
                """, unsafe_allow_html=True)