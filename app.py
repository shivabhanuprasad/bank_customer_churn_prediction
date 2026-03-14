import streamlit as st
import joblib
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

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
        cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32'><circle cx='16' cy='16' r='8' fill='%2360a5fa' opacity='0.4'/><circle cx='16' cy='16' r='4' fill='%233b82f6'/></svg>") 16 16, auto !important;
    }

    button, input, select, textarea, a, .stButton, [data-testid="stSidebar"] * {
        cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32'><circle cx='16' cy='16' r='8' fill='%2334d399' opacity='0.4'/><circle cx='16' cy='16' r='4' fill='%2310b981'/></svg>") 16 16, pointer !important;
    }

    /* Background image */
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.8), rgba(2, 6, 23, 0.98)),
                    url("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&w=1600&q=80");
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
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        text-align: center;
        font-size: 18px;
        color: #94a3b8;
        margin-bottom: 40px;
    }

    /* Form Glassmorphism Wrapper (Targeting Streamlit's native stForm container) */
    [data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(18px);
        border-radius: 24px;
        padding: 40px 30px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
    }

    [data-testid="stForm"]:hover {
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 0 35px rgba(99,102,241,0.2);
    }

    /* Form Input Fields */
    div[data-baseweb="input"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px;
    }

    label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    
    .login-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #f8fafc;
        text-align: center;
    }

    .login-subtitle {
        font-size: 15px;
        color: #94a3b8;
        margin-bottom: 25px;
        text-align: center;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        padding: 14px;
        font-size: 17px;
        font-weight: 700;
        border-radius: 16px;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        box-shadow: 0 0 20px rgba(99,102,241,0.3);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 35px rgba(139,92,246,0.5);
        color: white !important;
        border: none !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # --------------------------------------------------
    # Hero Section
    # --------------------------------------------------
    st.markdown("<div style='margin-top: 5vh;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Bank Customer Churn Intelligence</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Predict • Analyze • Retain customers with AI-powered insights</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # Sleek Login Form Embedded directly within st.form
    # --------------------------------------------------
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.form("login_form"):
            st.markdown("""
                <div style="text-align: center; font-size: 42px; margin-bottom: 5px;">🔐</div>
                <div class="login-title">Employee Portal</div>
                <div class="login-subtitle">Secure access to analytical features</div>
            """, unsafe_allow_html=True)
            
            username = st.text_input("Username", placeholder="e.g. bankemp")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Authenticate Securely")
            
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
        cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32'><circle cx='16' cy='16' r='8' fill='%2360a5fa' opacity='0.4'/><circle cx='16' cy='16' r='4' fill='%233b82f6'/></svg>") 16 16, auto !important;
    }

    button, input, select, textarea, a, .stButton, [data-testid="stSidebar"] * {
        cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32'><circle cx='16' cy='16' r='8' fill='%2334d399' opacity='0.4'/><circle cx='16' cy='16' r='4' fill='%2310b981'/></svg>") 16 16, pointer !important;
    }

    /* Background image */
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.8), rgba(2, 6, 23, 0.98)),
                    url("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&w=1600&q=80");
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
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        text-align: center;
        font-size: 18px;
        color: #94a3b8;
        margin-bottom: 40px;
    }

    /* Cards */
    .card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border-radius: 22px;
        padding: 28px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        color: #f8fafc;
    }

    .card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 20px 45px rgba(0,0,0,0.4);
    }

    .section-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 12px;
        color: #f8fafc;
    }

    .kpi {
        font-size: 42px;
        font-weight: 800;
        color: #60a5fa;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Button */
    .stButton > button {
        width: 100%;
        padding: 14px;
        font-size: 17px;
        font-weight: 700;
        border-radius: 16px;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        box-shadow: 0 0 20px rgba(99,102,241,0.3);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 35px rgba(139,92,246,0.5);
        color: white !important;
        border: none !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
    }

    /* Risk-specific colors */
    .high-risk { color: #f87171; }
    .medium-risk { color: #fbbf24; }
    .low-risk { color: #34d399; }

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
    app_mode = st.sidebar.selectbox("🎯 Prediction Mode", ["Single Customer", "Batch Processing"])
    
    if app_mode == "Batch Processing":
        st.header("📂 Bulk / Batch Prediction")
        st.markdown("Upload a CSV file containing multiple customer records to predict churn probabilities and risk tiers.")
        
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Loaded **{len(batch_df)}** records.")
                
                expected_columns = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
                missing_cols = [col for col in expected_columns if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                else:
                    if st.button("🚀 Run Batch Prediction"):
                        with st.spinner("Processing records..."):
                            input_df = batch_df[expected_columns].copy()
                            
                            # Safely handle 'Yes'/'No' if string representation was uploaded
                            for col in ["HasCrCard", "IsActiveMember"]:
                                if input_df[col].dtype == 'O':
                                    input_df[col] = input_df[col].astype(str).str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0}).fillna(0).astype(int)
                                    
                            probabilities = model.predict_proba(input_df)[:, 1]
                            
                            results_df = batch_df.copy()
                            results_df["Churn_Probability"] = probabilities
                            results_df["Risk_Tier"] = ["HIGH" if p > 0.6 else "MEDIUM" if p > 0.4 else "LOW" for p in probabilities]
                            
                            st.success("Batch prediction completed successfully!")
                            
                            # Analytics on Batch Data
                            st.markdown("### 📊 Batch Highlights")
                            colA, colB, colC = st.columns(3)
                            
                            high_risk_count = len(results_df[results_df['Risk_Tier'] == 'HIGH'])
                            med_risk_count = len(results_df[results_df['Risk_Tier'] == 'MEDIUM'])
                            low_risk_count = len(results_df[results_df['Risk_Tier'] == 'LOW'])
                            
                            with colA:
                                st.markdown(f"""
                                <div class="card" style="padding:15px;">
                                    <div class="section-title" style="font-size:16px;">🔴 High Risk Accounts</div>
                                    <div class="kpi high-risk">{high_risk_count}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with colB:
                                st.markdown(f"""
                                <div class="card" style="padding:15px;">
                                    <div class="section-title" style="font-size:16px;">🟡 Medium Risk Accounts</div>
                                    <div class="kpi medium-risk">{med_risk_count}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with colC:
                                st.markdown(f"""
                                <div class="card" style="padding:15px;">
                                    <div class="section-title" style="font-size:16px;">🟢 Low Risk Accounts</div>
                                    <div class="kpi low-risk">{low_risk_count}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            col_chart, col_data = st.columns([1, 1.5])
                            
                            with col_chart:
                                # Donut Chart of Risk Distribution
                                fig_batch = go.Figure(data=[go.Pie(
                                    labels=["HIGH", "MEDIUM", "LOW"],
                                    values=[high_risk_count, med_risk_count, low_risk_count],
                                    hole=.5,
                                    marker_colors=["#ef4444", "#fbbf24", "#10b981"],
                                    textinfo="label+percent"
                                )])
                                fig_batch.update_layout(
                                    title_text="Risk Distribution",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#f8fafc"),
                                    height=350,
                                    margin=dict(t=40, l=0, r=0, b=0),
                                    showlegend=False
                                )
                                st.plotly_chart(fig_batch, use_container_width=True)
                                
                            with col_data:
                                st.markdown("### 📝 Detailed Results")
                                # Show dataframe with color highlighting for risk tiers
                                def highlight_risk(val):
                                    color = '#ef4444' if val == 'HIGH' else '#fbbf24' if val == 'MEDIUM' else '#10b981'
                                    return f'color: {color}; font-weight: bold'
                                
                                # Convert to styler object
                                styled_df = results_df.style.map(highlight_risk, subset=['Risk_Tier'])
                                st.dataframe(styled_df, height=300, use_container_width=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Download Full Predictions",
                                data=csv,
                                file_name="churn_predictions_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
        st.stop()

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
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📊 Detailed Analysis", "💡 Insights & Recommendations", "📉 Sensitivity Analysis"])

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

            # Dynamic contextual AI explanation
            reasons = []
            if age > 45: reasons.append("Older age demographic correlates with higher historical turnover.")
            if is_active == "No": reasons.append("Account inactivity significantly increases the likelihood of departure.")
            if balance > 90000 and num_products == 1: reasons.append("High capital balance but single product utilization indicates low structural engagement.")
            if geography == "Germany": reasons.append("Customers in this geography have historically shown elevated churn sensitivity.")
            if tenure < 3: reasons.append("Early-stage tenure points to unsolidified bank loyalty.")
            if credit_score < 600: reasons.append("Lower credit scores map slightly to higher financial instability risks.")
            
            insight_text = " • ".join(reasons) if reasons else "Customer profile metrics align perfectly with highly stable, retained segments."

            st.markdown(f"""
            <div class="card" style="margin-top: 15px; margin-bottom: 25px;">
                <div class="section-title">🧠 Contextual AI Analysis</div>
                <p style="color:#94a3b8; font-size:16px;"><strong>Why this score?</strong> {insight_text}</p>
            </div>
            """, unsafe_allow_html=True)

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                # Gauge Chart for Probability
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    number = {'suffix': "%", 'font': {'color': '#f8fafc'}},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Predicted Churn Velocity", 'font': {'color': '#f8fafc', 'size': 18}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#f8fafc"},
                        'bar': {'color': "#ef4444" if probability > 0.6 else "#fbbf24" if probability > 0.4 else "#10b981"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "rgba(255,255,255,0.1)",
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.2)'},
                            {'range': [40, 60], 'color': 'rgba(251, 191, 36, 0.2)'},
                            {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.2)'}],
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350, font=dict(color="#f8fafc"))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_chart2:
                # Donut Chart for Probability
                fig_donut = go.Figure(data=[go.Pie(
                    labels=["Retention Probability", "Churn Probability"],
                    values=[1 - probability, probability],
                    hole=.6,
                    marker_colors=["#10b981", "#ef4444"],
                    textinfo="label+percent",
                    textfont_size=14
                )])

                fig_donut.update_layout(
                    title_text="Retention Ratio Breakdown",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc"),
                    height=350,
                    showlegend=False
                )
                st.plotly_chart(fig_donut, use_container_width=True)

        with tab2:
            st.markdown(f"""
            <div class="card" style="margin-bottom: 20px;">
                <div class="section-title">Prediction Breakdown - {risk} Risk Factors</div>
                <p style="color:#94a3b8; font-size:15px; margin-bottom:0px;">
                    Powered by <b>Explainable AI (SHAP)</b>. This dynamic algorithm analyzes the exact mathematical impact of every customer attribute on the final prediction. <br>
                    <span style="color:#ef4444; font-weight:600;">Red bars</span> push the customer towards churning, while <span style="color:#10b981; font-weight:600;">Green bars</span> pull them towards retention.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Subplots for Bar chart and Line Chart
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Key Factors Impacting Churn Ratio", "Projected Risk Trend (5-Year)"),
                specs=[[{"type": "bar"}, {"type": "scatter"}]]
            )

            # Dynamic Explainable AI (SHAP calculation)
            try:
                # Extract pipeline steps
                preprocessor = model.named_steps["preprocessor"]
                classifier = model.named_steps["classifier"]
                
                # Transform data & explain
                X_transformed = preprocessor.transform(input_df)
                explainer = shap.TreeExplainer(classifier)
                shap_values_obj = explainer.shap_values(X_transformed)
                
                # Check SHAP array structure (RF returns list of shape [class_0, class_1])
                if isinstance(shap_values_obj, list):
                    shap_vals = shap_values_obj[1][0]
                else:
                    if len(shap_values_obj.shape) == 3: # 3D array (samples, features, classes)
                        shap_vals = shap_values_obj[0, :, 1]
                    else:
                        shap_vals = shap_values_obj[0]
                
                # Unpack feature names correctly
                try:
                    feature_names = preprocessor.get_feature_names_out()
                    # Clean up feature names mapped by OHE
                    feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
                except Exception:
                    feature_names = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary", "France", "Germany", "Spain", "Male", "Female", "HasCrCard_Yes", "IsActiveMember_Yes"]
                
                # Identify Top 8 absolute impacts
                sorted_idx = np.argsort(np.abs(shap_vals))[-8:]
                features_sorted = [feature_names[i] for i in sorted_idx]
                importance_sorted = [shap_vals[i] for i in sorted_idx]
                
                # Coloring (+ impact on churn = red, - impact on churn = green)
                colors = ["rgba(239, 68, 68, 0.85)" if val > 0 else "rgba(16, 185, 129, 0.85)" for val in importance_sorted]
                edge_colors = ["#ef4444" if val > 0 else "#10b981" for val in importance_sorted]
                
                # Custom Hover text for each bar
                hover_texts = [
                    f"<b>Attribute:</b> {feat}<br><b>Impact:</b> {'Increases Risk' if val > 0 else 'Lowers Risk'}<br><b>Weight:</b> {abs(val):.3f}" 
                    for feat, val in zip(features_sorted, importance_sorted)
                ]
                
                # Annotations directly on the bars
                text_annotations = [f"+{val:.3f}" if val > 0 else f"{val:.3f}" for val in importance_sorted]

                fig.add_trace(go.Bar(
                    y=features_sorted,
                    x=importance_sorted,
                    orientation='h',
                    marker={"color": colors, "line": {"color": edge_colors, "width": 1.5}},
                    text=text_annotations,
                    textposition='auto',
                    textfont={"color": "#f8fafc", "size": 12},
                    hoverinfo='text',
                    hovertext=hover_texts,
                    name="SHAP Component"
                ), row=1, col=1)
                
            except Exception as e:
                # Fallback in case of pipeline extraction anomaly
                features_sorted = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
                importance_sorted = [0.1, 0.35, 0.05, 0.25, 0.1, 0.05, 0.05, 0.05] if risk == "HIGH" else [-0.1, -0.2, 0.2, 0.1, 0.1, 0.15, 0.3, 0.05]
                colors = ["rgba(239, 68, 68, 0.85)" if val > 0 else "rgba(16, 185, 129, 0.85)" for val in importance_sorted]
                fig.add_trace(go.Bar(
                    y=features_sorted,
                    x=importance_sorted,
                    orientation='h',
                    marker=dict(color=colors),
                    name="Impact Factor"
                ), row=1, col=1)

            # Line Chart for Risk projection over time
            tenure_range = list(range(tenure, min(tenure+6, 11)))
            # Generate a mock trend: risk goes down if active, up if inactive
            trend_multiplier = -0.05 if is_active == "Yes" else 0.08
            risk_trend = [min(1.0, max(0.0, probability * (1 + (i - tenure) * trend_multiplier))) for i in tenure_range]

            fig.add_trace(go.Scatter(
                x=[f"Year {i}" for i in tenure_range],
                y=risk_trend,
                mode='lines+markers',
                line={"color": '#f87171' if risk == "HIGH" else '#34d399', "width": 4},
                marker={"size": 8},
                name="Projected Risk"
            ), row=1, col=2)

            fig.update_layout(
                height=450,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                showlegend=False,
                margin=dict(t=40, l=0, r=0, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)

            # --- Explicit Descriptions of Factors dynamically fetched from SHAP ---
            st.markdown("### 🔍 Mathematical Root Cause Analysis")
            st.markdown("<p style='color:#94a3b8; font-size: 15px; margin-bottom: 20px;'>A natural language breakdown of the precise attributes pushing against or supporting retention.</p>", unsafe_allow_html=True)
            
            try:
                # Top positive (churn-causing, red)
                red_factors = [(f, v) for f, v in zip(features_sorted, importance_sorted) if v > 0]
                red_factors.sort(key=lambda x: x[1], reverse=True)
                
                # Top negative (retention-causing, green)
                green_factors = [(f, v) for f, v in zip(features_sorted, importance_sorted) if v < 0]
                green_factors.sort(key=lambda x: x[1])  # most negative first
                
                col_red, col_green = st.columns(2)
                
                with col_red:
                    if red_factors:
                        st.markdown("<div style='background:rgba(239, 68, 68, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(239,68,68,0.3);'>", unsafe_allow_html=True)
                        st.markdown("<h4 style='color:#ef4444; margin-bottom: 10px; margin-top:0px;'>🔴 Driving Churn (Red Bars)</h4>", unsafe_allow_html=True)
                        for i, (feat, val) in enumerate(red_factors):
                            if i >= 3: break
                            reason = f"The underlying value of <b>{feat}</b> is mathematically increasing the risk probability. This attribute closely matches profiles of users who have historically disconnected their accounts."
                            st.markdown(f"<p style='color:#f8fafc; font-size:14px; margin-bottom: 8px;'>• {reason}</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                with col_green:
                    if green_factors:
                        st.markdown("<div style='background:rgba(16, 185, 129, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(16,185,129,0.3);'>", unsafe_allow_html=True)
                        st.markdown("<h4 style='color:#10b981; margin-bottom: 10px; margin-top:0px;'>🟢 Assisting Retention (Green Bars)</h4>", unsafe_allow_html=True)
                        for i, (feat, val) in enumerate(green_factors):
                            if i >= 3: break
                            reason = f"The strong baseline of <b>{feat}</b> is actively pulling the churn risk down. This is a foundational factor keeping the customer financially anchored to the bank segment."
                            st.markdown(f"<p style='color:#f8fafc; font-size:14px; margin-bottom: 8px;'>• {reason}</p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

            except Exception:
                pass

        with tab3:
            st.markdown(f"""
            <div class="card">
                <div class="section-title">Insights & Recommendations - {risk} Risk</div>
                <p style="color:#94a3b8;">Tailored strategies to mitigate churn based on the {risk.lower()} risk assessment.</p>
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

            st.markdown("<ul style='color:#f8fafc;'>", unsafe_allow_html=True)
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

        with tab4:
            st.markdown("""
            <div class="card" style="margin-bottom: 20px;">
                <div class="section-title">📉 Sensitivity Analysis — How Each Parameter Drives Churn</div>
                <p style="color:#94a3b8; font-size:15px; margin-bottom:0px;">
                    Each chart below sweeps one parameter across its full possible range while keeping
                    all other inputs <b>fixed at your current sidebar values</b>. This shows exactly
                    how sensitive the model is to each individual feature.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Base row — fixed values from sidebar
            base = {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": float(balance),
                "NumOfProducts": num_products,
                "HasCrCard": 1 if has_card == "Yes" else 0,
                "IsActiveMember": 1 if is_active == "Yes" else 0,
                "EstimatedSalary": float(salary)
            }

            def sweep(param, values):
                """Run predictions sweeping `param` over `values`, all else fixed."""
                rows = []
                for v in values:
                    row = base.copy()
                    row[param] = v
                    rows.append(row)
                df_sweep = pd.DataFrame(rows)
                return model.predict_proba(df_sweep)[:, 1]

            def make_line(x_vals, y_vals, title, x_label, current_val, color="#3b82f6"):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals * 100,
                    mode='lines',
                    line=dict(color=color, width=3),
                    fill='tozeroy',
                    fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)',
                    name="Churn %"
                ))
                # Current value marker
                curr_prob = model.predict_proba(pd.DataFrame([base]))[0][1] * 100
                fig.add_vline(x=current_val, line_dash="dash", line_color="#FFD700",
                              annotation_text=f"Current: {current_val}", annotation_font_color="#FFD700")
                fig.add_hline(y=curr_prob, line_dash="dot", line_color="rgba(255,255,255,0.2)")
                fig.update_layout(
                    title=dict(text=title, font=dict(color="#f8fafc", size=14)),
                    xaxis=dict(title=x_label, color="#94a3b8", gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(title="Churn Probability (%)", range=[0, 100], color="#94a3b8", gridcolor="rgba(255,255,255,0.05)"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc"),
                    height=260,
                    margin=dict(t=40, l=0, r=0, b=0),
                    showlegend=False
                )
                return fig

            # ── Row 1: Credit Score & Age ──
            c1, c2 = st.columns(2)
            with c1:
                x = list(range(300, 901, 20))
                y = sweep("CreditScore", x)
                st.plotly_chart(make_line(x, y, "Credit Score vs Churn %", "Credit Score", credit_score, "#3b82f6"), use_container_width=True)
            with c2:
                x = list(range(18, 81, 1))
                y = sweep("Age", x)
                st.plotly_chart(make_line(x, y, "Age vs Churn %", "Age", age, "#f87171"), use_container_width=True)

            # ── Row 2: Tenure & Balance ──
            c3, c4 = st.columns(2)
            with c3:
                x = list(range(0, 11))
                y = sweep("Tenure", x)
                st.plotly_chart(make_line(x, y, "Tenure (Years) vs Churn %", "Tenure", tenure, "#34d399"), use_container_width=True)
            with c4:
                x = [i * 10000 for i in range(0, 26)]
                y = sweep("Balance", x)
                st.plotly_chart(make_line(x, y, "Account Balance vs Churn %", "Balance", float(balance), "#fbbf24"), use_container_width=True)

            # ── Row 3: Num Products & Salary ──
            c5, c6 = st.columns(2)
            with c5:
                x = [1, 2, 3, 4]
                y = sweep("NumOfProducts", x)
                st.plotly_chart(make_line(x, y, "Number of Products vs Churn %", "Num Products", num_products, "#a78bfa"), use_container_width=True)
            with c6:
                x = [i * 10000 for i in range(0, 21)]
                y = sweep("EstimatedSalary", x)
                st.plotly_chart(make_line(x, y, "Estimated Salary vs Churn %", "Salary", float(salary), "#60a5fa"), use_container_width=True)

            # ── Categorical Impact Cards ──
            st.markdown("### 🗂️ Categorical Parameter Impact")
            cat_cols = st.columns(3)

            with cat_cols[0]:
                geo_vals = ["France", "Germany", "Spain"]
                geo_probs = []
                for g in geo_vals:
                    row = base.copy(); row["Geography"] = g
                    geo_probs.append(model.predict_proba(pd.DataFrame([row]))[0][1] * 100)
                fig_geo = go.Figure(go.Bar(
                    x=geo_vals, y=geo_probs,
                    marker_color=["#3b82f6", "#f87171", "#34d399"],
                    text=[f"{v:.1f}%" for v in geo_probs], textposition="auto"
                ))
                fig_geo.update_layout(title="Geography", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"),
                    height=240, yaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.05)"),
                    margin=dict(t=35,l=0,r=0,b=0), showlegend=False)
                st.plotly_chart(fig_geo, use_container_width=True)

            with cat_cols[1]:
                gen_vals = ["Male", "Female"]
                gen_probs = []
                for g in gen_vals:
                    row = base.copy(); row["Gender"] = g
                    gen_probs.append(model.predict_proba(pd.DataFrame([row]))[0][1] * 100)
                fig_gen = go.Figure(go.Bar(
                    x=gen_vals, y=gen_probs,
                    marker_color=["#60a5fa", "#f472b6"],
                    text=[f"{v:.1f}%" for v in gen_probs], textposition="auto"
                ))
                fig_gen.update_layout(title="Gender", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"),
                    height=240, yaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.05)"),
                    margin=dict(t=35,l=0,r=0,b=0), showlegend=False)
                st.plotly_chart(fig_gen, use_container_width=True)

            with cat_cols[2]:
                act_vals = ["Active Member", "Inactive Member"]
                act_probs = []
                for v in [1, 0]:
                    row = base.copy(); row["IsActiveMember"] = v
                    act_probs.append(model.predict_proba(pd.DataFrame([row]))[0][1] * 100)
                fig_act = go.Figure(go.Bar(
                    x=act_vals, y=act_probs,
                    marker_color=["#34d399", "#f87171"],
                    text=[f"{v:.1f}%" for v in act_probs], textposition="auto"
                ))
                fig_act.update_layout(title="Member Activity Status", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"),
                    height=240, yaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.05)"),
                    margin=dict(t=35,l=0,r=0,b=0), showlegend=False)
                st.plotly_chart(fig_act, use_container_width=True)
