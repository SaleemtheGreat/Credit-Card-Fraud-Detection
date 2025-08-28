import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ===============================
# Load dataset (for EDA)
# ===============================
DATA_PATH = os.path.join("data", "creditcard.csv")
df = pd.read_csv(DATA_PATH)

# ===============================
# Load trained model
# ===============================
MODEL_PATH = os.path.join("models", "rf.joblib")
model = joblib.load(MODEL_PATH)

# ===============================
# Streamlit App
# ===============================
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")
st.markdown("Banks lose millions due to fraudulent transactions. This app demonstrates **real-time fraud detection** using ML models.")

# ===============================
# Sidebar Navigation
# ===============================
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to:", ["EDA", "Model Prediction", "About"])

# ===============================
# EDA Page
# ===============================
if options == "EDA":
    st.header("Exploratory Data Analysis (EDA)")
    st.subheader("1. Sample Data")
    st.write(df.head())
    
    st.subheader("2. Class Distribution")
    class_counts = df['Class'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
    ax.set_xticklabels(["Legit (0)", "Fraud (1)"])
    ax.set_ylabel("Count")
    st.pyplot(fig)
    
    st.subheader("3. Transaction Amount Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Amount'], bins=50, ax=ax, log_scale=True)
    st.pyplot(fig)

    st.subheader("4. Fraud vs Non-Fraud Amount")
    fig, ax = plt.subplots()
    sns.boxplot(x="Class", y="Amount", data=df, ax=ax)
    ax.set_xticklabels(["Legit (0)", "Fraud (1)"])
    st.pyplot(fig)

# ===============================
# Model Prediction Page
# ===============================
elif options == "Model Prediction":
    st.header("🔎 Real-Time Fraud Detection")
    st.markdown("Enter transaction details manually or test with example data.")

    # Initialize session state to hold current feature values
    if 'current_features' not in st.session_state:
        st.session_state.current_features = [0.0]*30  # Time + V1..V28 + Amount

    # --- Example Testing ---
    st.subheader("Test with Example Transactions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Load Random Fraud Example"):
            sample = df[df['Class'] == 1].sample(1).iloc[0]
            st.session_state.current_features = sample.drop("Class").values.tolist()
            st.warning("⚠️ Random Fraud Example Loaded")

    with col2:
        if st.button("Load Random Legit Example"):
            sample = df[df['Class'] == 0].sample(1).iloc[0]
            st.session_state.current_features = sample.drop("Class").values.tolist()
            st.success("✅ Random Legit Example Loaded")

    # --- Manual / Dynamic Input ---
    st.subheader("Transaction Features")
    
    # Time input
    st.session_state.current_features[0] = st.number_input("Transaction Time", value=float(st.session_state.current_features[0]))

    # V1 - V28 inputs
    v_features = []
    cols = st.columns(4)
    for i in range(1, 29):
        with cols[(i - 1) % 4]:
            val = st.number_input(f"V{i}", value=float(st.session_state.current_features[i]))
            st.session_state.current_features[i] = val
            v_features.append(val)

    # Amount input
    st.session_state.current_features[29] = st.number_input("Transaction Amount", value=float(st.session_state.current_features[29]))

    # Predict button
    if st.button("Predict Transaction"):
        X = np.array(st.session_state.current_features).reshape(1, -1)
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None

        if prediction == 1:
            st.error("⚠️ Suspicious Transaction Detected! FRAUD")
        else:
            st.success("✅ Transaction is Safe.")

        if proba is not None:
            st.info(f"Fraud Probability: {proba:.2f}")

# ===============================
# About Page
# ===============================
else:
    st.header("ℹ️ About")
    st.write("""
    **Project Workflow:**
    1. **Dataset:** Kaggle’s Credit Card Fraud Dataset  
    2. **EDA:** Visualization of transactions, fraud distribution  
    3. **Preprocessing:** Class imbalance handled with SMOTE/undersampling  
    4. **Models Used:** Logistic Regression, Random Forest, Isolation Forest, Autoencoders  
    5. **Evaluation Metrics:** Precision, Recall, F1-score (better than plain accuracy)  
    6. **Deployment:** This Streamlit app for real-time fraud alerts 🚀  
    """)
