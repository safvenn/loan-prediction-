"""
Streamlit Loan Prediction App - Stylish Version
A bold, distinctive UI for loan status prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Loan Predictor Pro",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - Bold Aesthetic Design
# ============================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');
    
    :root {
        --primary: #0D0D0D;
        --secondary: #1A1A2E;
        --accent: #E94560;
        --accent-glow: #FF6B6B;
        --success: #00F5D4;
        --warning: #FEE440;
        --card-bg: #16213E;
        --text-primary: #FFFFFF;
        --text-secondary: #A0A0A0;
    }
    
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0D0D0D 0%, #1A1A2E 50%, #0F0F23 100%);
        min-height: 100vh;
    }
    
    /* Main Title */
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        background: linear-gradient(90deg, #E94560, #FF6B6B, #E94560);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
    }
    
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    
    .subtitle {
        font-family: 'Space Mono', monospace;
        color: #A0A0A0;
        text-align: center;
        font-size: 0.9rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .custom-card {
        background: linear-gradient(145deg, #16213E 0%, #1A1A2E 100%);
        border: 1px solid rgba(233, 69, 96, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-4px);
        border-color: rgba(233, 69, 96, 0.5);
        box-shadow: 0 12px 40px rgba(233, 69, 96, 0.15),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #1A1A2E 0%, #0D0D0D 100%);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid rgba(0, 245, 212, 0.3);
        box-shadow: 0 4px 20px rgba(0, 245, 212, 0.1);
    }
    
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #00F5D4;
    }
    
    .metric-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #A0A0A0;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.5rem;
    }
    
    /* Input Fields */
    .stSelectbox, .stNumberInput, .stTextInput {
        background: #16213E !important;
        border: 1px solid rgba(233, 69, 96, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox > div > div, .stNumberInput > div > div, .stTextInput > div > div {
        background: transparent !important;
        color: #FFFFFF !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #E94560 0%, #FF6B6B 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        color: white !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(233, 69, 96, 0.6) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D0D0D 0%, #1A1A2E 100%) !important;
        border-right: 1px solid rgba(233, 69, 96, 0.2) !important;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #E94560 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(22, 33, 62, 0.8);
        border: 1px solid rgba(233, 69, 96, 0.3);
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        color: #A0A0A0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #E94560 0%, #FF6B6B 100%) !important;
        color: white !important;
        border-color: #E94560 !important;
    }
    
    /* DataFrame */
    .stDataFrame {
        border: 1px solid rgba(233, 69, 96, 0.2) !important;
        border-radius: 12px !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(0, 245, 212, 0.1) !important;
        border: 1px solid #00F5D4 !important;
        border-radius: 8px !important;
    }
    
    .stError {
        background: rgba(233, 69, 96, 0.1) !important;
        border: 1px solid #E94560 !important;
        border-radius: 8px !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(233, 69, 96, 0.5), transparent);
        margin: 2rem 0;
    }
    
    /* Prediction Result */
    .prediction-result {
        font-family: 'Outfit', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
    }
    
    .prediction-approved {
        background: linear-gradient(135deg, rgba(0, 245, 212, 0.2) 0%, rgba(0, 245, 212, 0.1) 100%);
        border: 2px solid #00F5D4;
        color: #00F5D4;
    }
    
    .prediction-rejected {
        background: linear-gradient(135deg, rgba(233, 69, 96, 0.2) 0%, rgba(233, 69, 96, 0.1) 100%);
        border: 2px solid #E94560;
        color: #E94560;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: fadeInUp 0.5s ease forwards;
    }
    
    .delay-1 { animation-delay: 0.1s; }
    .delay-2 { animation-delay: 0.2s; }
    .delay-3 { animation-delay: 0.3s; }
    .delay-4 { animation-delay: 0.4s; }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD AND PREPARE DATA
# ============================================
import os

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "loan_data.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def train_models(x_train, y_train):
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(x_train, y_train)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)
    
    return lr_model, rf_model

def preprocess_data(df):
    le = LabelEncoder()
    categorical_columns = ['person_gender', 'person_home_ownership', 'loan_intent', 
                          'person_education', 'previous_loan_defaults_on_file']
    
    df_processed = df.copy()
    for col in categorical_columns:
        df_processed[col] = le.fit_transform(df_processed[col])
    
    return df_processed


# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("---")
    st.markdown("#### 🎯 Model Selection")
    model_choice = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Random Forest"],
        index=1
    )
    
    st.markdown("---")
    st.markdown("#### 📊 Data Overview")
    st.info("💡 This app predicts loan approval status based on applicant details.")
    
    st.markdown("---")
    st.markdown("#### 🔧 Settings")
    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    random_state = st.number_input("Random State", 0, 100, 42)

# ============================================
# MAIN CONTENT
# ============================================

# Title
st.markdown('<h1 class="main-title">LOAN PREDICTOR PRO</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">✨ AI-Powered Loan Approval Prediction ✨</p>', unsafe_allow_html=True)

# Load and prepare data
df = load_data()
df_processed = preprocess_data(df)

# Feature and target
x = df_processed.drop('loan_status', axis=1)
y = df_processed['loan_status']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

# Train models
lr_model, rf_model = train_models(x_train, y_train)

# Select model based on choice
if model_choice == "Logistic Regression":
    model = lr_model
    model_name = "Logistic Regression"
else:
    model = rf_model
    model_name = "Random Forest"

# Get predictions
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

# ============================================
# TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📈 Analytics", "🔮 Predict", "📋 Data"])

# ---------- TAB 1: HOME ----------
with tab1:
    # Hero Section with Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card animate-in">
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-label">Total Records</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card animate-in delay-1">
                <div class="metric-value">{len(x.columns)}</div>
                <div class="metric-label">Features</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card animate-in delay-2">
                <div class="metric-value">{accuracy:.1%}</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        approved = (y == 1).sum()
        st.markdown(f"""
            <div class="metric-card animate-in delay-3">
                <div class="metric-value">{approved:,}</div>
                <div class="metric-label">Approved Loans</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Quick Stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="custom-card animate-in">
                <h3 style="color: #E94560; margin-bottom: 1rem;">🎯 How It Works</h3>
                <p style="color: #A0A0A0; line-height: 1.8;">
                    Our machine learning model analyzes applicant data to predict loan approval status.
                    Simply enter the applicant details in the <strong style="color: #00F5D4;">Predict</strong> tab
                    and get instant predictions!
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="custom-card animate-in delay-1">
                <h3 style="color: #E94560; margin-bottom: 1rem;">💎 Features</h3>
                <ul style="color: #A0A0A0; line-height: 1.8; list-style: none; padding: 0;">
                    <li>✓ Real-time predictions</li>
                    <li>✓ Multiple ML models</li>
                    <li>✓ Detailed analytics</li>
                    <li>✓ Interactive visualizations</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ---------- TAB 2: ANALYTICS ----------
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="custom-card">
                <h3 style="color: #E94560;">📊 Model Performance</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Accuracy comparison
        lr_acc = accuracy_score(y_test, lr_model.predict(x_test))
        rf_acc = accuracy_score(y_test, rf_model.predict(x_test))
        
        st.markdown(f"""
            <div style="display: flex; justify-content: space-around; margin: 1rem 0;">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #00F5D4;">{lr_acc:.1%}</div>
                    <div style="color: #A0A0A0; font-size: 0.8rem;">Logistic Regression</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #E94560;">{rf_acc:.1%}</div>
                    <div style="color: #A0A0A0; font-size: 0.8rem;">Random Forest</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="custom-card">
                <h3 style="color: #E94560;">📈 Loan Status Distribution</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Simple bar chart using markdown
        approved = (y == 1).sum()
        rejected = (y == 0).sum()
        total = len(y)
        
        st.markdown(f"""
            <div style="margin: 1rem 0;">
                <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                    <span style="color: #00F5D4; width: 80px;">Approved</span>
                    <div style="flex: 1; height: 20px; background: #1A1A2E; border-radius: 10px; overflow: hidden;">
                        <div style="width: {approved/total*100}%; height: 100%; background: linear-gradient(90deg, #00F5D4, #00D4AA);"></div>
                    </div>
                    <span style="color: #A0A0A0; margin-left: 10px;">{approved:,}</span>
                </div>
                <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                    <span style="color: #E94560; width: 80px;">Rejected</span>
                    <div style="flex: 1; height: 20px; background: #1A1A2E; border-radius: 10px; overflow: hidden;">
                        <div style="width: {rejected/total*100}%; height: 100%; background: linear-gradient(90deg, #E94560, #FF6B6B);"></div>
                    </div>
                    <span style="color: #A0A0A0; margin-left: 10px;">{rejected:,}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Classification Report
    st.markdown("""
        <div class="custom-card">
            <h3 style="color: #E94560;">📋 Classification Report</h3>
        </div>
    """, unsafe_allow_html=True)
    
    report = classification_report(y_test, y_pred, target_names=['Rejected', 'Approved'])
    st.code(report, language="text")

# ---------- TAB 3: PREDICT ----------
with tab3:
    st.markdown("""
        <div class="custom-card">
            <h3 style="color: #E94560; margin-bottom: 1.5rem;">🔮 Make a Prediction</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        person_age = st.number_input("Age", 18, 100, 30)
        person_income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
        person_emp_exp = st.number_input("Employment Experience (years)", 0, 50, 5)
        person_education = st.selectbox("Education", df['person_education'].unique())
        person_gender = st.selectbox("Gender", df['person_gender'].unique())
    
    with col2:
        person_home_ownership = st.selectbox("Home Ownership", df['person_home_ownership'].unique())
        loan_intent = st.selectbox("Loan Intent", df['loan_intent'].unique())
        # loan_grade removed - not in dataset
        loan_amnt = st.number_input("Loan Amount ($)", 0, 100000, 10000)
        loan_int_rate = st.number_input("Interest Rate (%)", 0.0, 50.0, 10.5)
    
    # Additional inputs
    col3, col4 = st.columns(2)
    
    with col3:
        loan_percent_income = st.number_input("Loan as % of Income", 0.0, 100.0, 20.0)
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", 0, 50, 5)
    
    with col4:
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        previous_loan_defaults = st.selectbox("Previous Loan Defaults", df['previous_loan_defaults_on_file'].unique())
    
    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 PREDICT LOAN STATUS", use_container_width=True):
        # Create input dataframe
        input_data = {
            'person_age': person_age,
            'person_income': person_income,
            'person_emp_exp': person_emp_exp,
            'person_education': person_education,
            'person_gender': person_gender,
            'person_home_ownership': person_home_ownership,
            'loan_intent': loan_intent,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'credit_score': credit_score,
            'previous_loan_defaults_on_file': previous_loan_defaults
        }
        
        # Create dataframe and encode
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical columns
        le = LabelEncoder()
        categorical_cols = ['person_gender', 'person_home_ownership', 'loan_intent', 
                           'person_education', 'previous_loan_defaults_on_file']
        
        for col in categorical_cols:
            input_df[col] = le.fit_transform(input_df[col].astype(str))
        
        # Ensure columns match training data
        for col in x.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[x.columns]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Display result
        if prediction == 1:
            st.markdown(f"""
                <div class="prediction-result prediction-approved animate-in">
                    ✓ APPROVED
                </div>
                <p style="text-align: center; color: #A0A0A0;">
                    Confidence: {probability[1]:.1%}
                </p>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="prediction-result prediction-rejected animate-in">
                    ✗ REJECTED
                </div>
                <p style="text-align: center; color: #A0A0A0;">
                    Confidence: {probability[0]:.1%}
                </p>
            """, unsafe_allow_html=True)

# ---------- TAB 4: DATA ----------
with tab4:
    st.markdown("""
        <div class="custom-card">
            <h3 style="color: #E94560;">📋 Dataset Overview</h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="custom-card">
                <h3 style="color: #E94560;">📊 Data Info</h3>
            </div>
        """, unsafe_allow_html=True)
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    with col2:
        st.markdown("""
            <div class="custom-card">
                <h3 style="color: #E94560;">📋 Column Types</h3>
            </div>
        """, unsafe_allow_html=True)
        st.write(df.dtypes.to_string())

# ============================================
# FOOTER
# ============================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <p style="text-align: center; color: #A0A0A0; font-family: 'Space Mono', monospace; font-size: 0.8rem;">
        💎 Loan Predictor Pro | Built with Streamlit & Scikit-Learn 💎
    </p>
""", unsafe_allow_html=True)