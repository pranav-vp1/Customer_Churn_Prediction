"""
app.py  —  Customer Churn Prediction  |  Streamlit Deployment
Run:  streamlit run app.py
"""
# python -m venv venv
# venv\Scripts\activate
# pip install -r requirements.txt
# streamlit run app.py
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()
model            = artifacts["model"]
scaler           = artifacts["scaler"]
feature_columns  = artifacts["feature_columns"]
label_encoders   = artifacts["label_encoders"]
multi_cols       = artifacts["multi_cols"]
THRESHOLD        = artifacts["threshold"]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 Customer Churn Prediction")
st.markdown(
    """
    **ML Model:** Tuned Gradient Boosting Classifier &nbsp;|&nbsp;
    **Dataset:** Telecom Customer Data &nbsp;|&nbsp;
    **Threshold:** 0.40 (optimised for Recall)
    """
)
st.divider()

# ── Sidebar – about ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About This App")
    st.markdown(
        """
        This app predicts whether a telecom customer is likely to **churn**
        (cancel their subscription) based on account and service details.

        **Top Churn Drivers:**
        - Short tenure
        - High monthly charges
        - Month-to-month contract
        - No tech support
        - Low total charges (new customers)

        **Business Recommendations:**
        - Offer long-term contract discounts
        - Bundle tech support in base plans
        - Flag customers with tenure < 12 months
        - Introduce loyalty pricing above $70/month
        """
    )
    st.divider()
    st.caption("Built with Scikit-learn & Streamlit by Pranav V P · Internship Project")

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("🔢 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    gender           = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen   = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner          = st.selectbox("Partner", ["Yes", "No"])
    dependents       = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    st.markdown("**Account Info**")
    tenure           = st.slider("Tenure (months)", 0, 72, 12)
    contract         = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method   = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

with col3:
    st.markdown("**Charges**")
    monthly_charges  = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
    total_charges    = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0, step=10.0)

st.divider()
st.subheader("📶 Services")

scol1, scol2, scol3, scol4 = st.columns(4)

with scol1:
    phone_service    = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines   = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

with scol2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security  = st.selectbox("Online Security", ["No internet service", "No", "Yes"])

with scol3:
    online_backup    = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])

with scol4:
    tech_support     = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    streaming_tv     = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

# ── Prediction logic ──────────────────────────────────────────────────────────
def build_input_df():
    """Build a raw single-row dataframe matching notebook's pre-encoding format."""
    raw = {
        "SeniorCitizen":    senior_citizen,
        "tenure":           tenure,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
        "gender":           gender,
        "Partner":          partner,
        "Dependents":       dependents,
        "PhoneService":     phone_service,
        "PaperlessBilling": paperless_billing,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet_service,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_protection,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "Contract":         contract,
        "PaymentMethod":    payment_method,
    }
    return pd.DataFrame([raw])


def preprocess(df_raw):
    """Apply the same encoding + scaling pipeline used during training."""
    df = df_raw.copy()

    # IQR clipping (use same bounds — approximate with training stats)
    # The scaler handles normalisation; clipping extremes is optional for
    # single-row inference but keeps it consistent.
    df["tenure"]         = df["tenure"].clip(0, 72)
    df["MonthlyCharges"] = df["MonthlyCharges"].clip(18.0, 118.0)
    df["TotalCharges"]   = df["TotalCharges"].clip(0, 8670.0)

    # Label encode binary columns
    binary_cols_order = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols_order:
        le = label_encoders[col]
        df[col] = le.transform(df[col])

    # One-hot encode multi-class columns
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # Align to training feature columns (fill any missing dummies with 0)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    # Scale
    X_scaled = scaler.transform(df)
    return X_scaled


st.divider()
predict_btn = st.button("🔍 Predict Churn", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Running prediction…"):
        try:
            df_raw   = build_input_df()
            X_scaled = preprocess(df_raw)

            prob      = model.predict_proba(X_scaled)[0][1]
            prediction = int(prob >= THRESHOLD)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

    st.divider()
    st.subheader("📊 Prediction Result")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 1:
            st.error("### 🚨 HIGH CHURN RISK")
            st.markdown(
                f"This customer is **likely to churn**.\n\n"
                f"Consider proactive retention measures immediately."
            )
        else:
            st.success("### ✅ LOW CHURN RISK")
            st.markdown(
                f"This customer is **likely to stay**.\n\n"
                f"Continue monitoring their satisfaction."
            )

    with res_col2:
        st.metric("Churn Probability", f"{prob * 100:.1f}%")
        st.metric("Decision Threshold", f"{THRESHOLD * 100:.0f}%")
        st.progress(float(prob))

    # Risk breakdown
    st.divider()
    st.subheader("🧩 Key Risk Factors for This Customer")

    risk_notes = []
    if tenure < 12:
        risk_notes.append("⚠️ **Short tenure** (< 12 months) — new customers are higher risk.")
    if monthly_charges > 70:
        risk_notes.append("⚠️ **High monthly charges** (> $70) — correlates with churn.")
    if contract == "Month-to-month":
        risk_notes.append("⚠️ **Month-to-month contract** — highest churn rate segment.")
    if tech_support == "No":
        risk_notes.append("⚠️ **No Tech Support** — significantly increases churn likelihood.")
    if total_charges < 500:
        risk_notes.append("⚠️ **Low total charges** — indicates a newer, higher-risk customer.")

    if risk_notes:
        for note in risk_notes:
            st.markdown(note)
    else:
        st.markdown("✅ No major individual risk factors detected.")