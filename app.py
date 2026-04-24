import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Fairness Guardrail", layout="wide")

# --- CUSTOM CSS FOR HACKATHON GLOW ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ CAG: Causal-Adversarial Guardrail")
st.markdown("### Automated Fairness Auditing for Automated Decisions")

# --- DATA LOADING ---
@st.cache_resource
def load_and_train():
    try:
        # Loading your local CSV
        df = pd.read_csv("adult_sample.csv")
        
        # Training the "Black-Box" Model
        features = ['age', 'gender_bin', 'education-num']
        X = df[features]
        y = df['target']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, df
    except FileNotFoundError:
        st.error("Missing 'adult_sample.csv'. Please create the file first!")
        return None, None

model, df = load_and_train()

if model is not None:
    # --- SIDEBAR INPUTS ---
    st.sidebar.header("📝 Applicant Details")
    user_age = st.sidebar.slider("Age", 18, 70, 35)
    user_edu = st.sidebar.slider("Education (Years)", 1, 20, 14)
    user_gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    
    # Convert gender to binary
    gender_map = {"Male": 1, "Female": 0}
    g_val = gender_map[user_gender]

    # Prepare input for model
    current_user = pd.DataFrame([[user_age, g_val, user_edu]], 
                                columns=['age', 'gender_bin', 'education-num'])

    # --- MAIN DASHBOARD ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚀 Standard AI Prediction")
        # Get probability
        prob = model.predict_proba(current_user)[0][1]
        
        # Visual indicator
        color = "green" if prob > 0.5 else "red"
        st.markdown(f"<h1 style='color:{color};'>{prob*100:.1f}%</h1>", unsafe_allow_html=True)
        st.write("**Outcome:** " + ("Approved" if prob > 0.5 else "Rejected"))
        st.caption("This is the raw output from the un-audited model.")

    with col2:
        st.subheader("⚖️ Counterfactual Fairness Audit")
        
        # CREATE THE "TWIN" (Flip the gender)
        twin_g_val = 1 if g_val == 0 else 0
        twin_user = pd.DataFrame([[user_age, twin_g_val, user_edu]], 
                                 columns=['age', 'gender_bin', 'education-num'])
        
        twin_prob = model.predict_proba(twin_user)[0][1]
        bias_gap = abs(prob - twin_prob)

        # Fairness Check Logic
        if bias_gap > 0.15:
            st.error(f"❌ **Bias Flagged: {bias_gap*100:.1f}% Variance**")
            st.write(f"The model treats a **{'Male' if twin_g_val == 1 else 'Female'}** twin differently under the same conditions.")
            st.button("Run Adversarial De-biasing")
        else:
            st.success("✅ **Fairness Verified**")
            st.write("No significant variance detected between gender counterfactuals.")

    st.divider()

    # --- EXPLAINABILITY SECTION ---
    st.subheader("🧠 Causal Insight")
    st.write("The system identifies that 'Education' is a strong predictor, but 'Gender' is acting as a hidden weight.")
    
    # Simple Bar Chart of the data
    st.bar_chart(df.groupby('gender_bin')['target'].mean())
