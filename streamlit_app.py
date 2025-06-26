import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model and scaler
MODEL_PATH = os.path.join('model.pkl')
SCALER_PATH = os.path.join('scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# UI
st.title("🛫 Travel Insurance Predictor")
st.write("Fill in the details to check if someone is likely to need travel insurance.")

with st.form("travel_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    employment_type = st.selectbox("Employment Type", ['Private Sector/Self Employed', 'Government Sector'])
    graduate = st.radio("Graduate?", ['Yes', 'No'])
    annual_income = st.number_input("Annual Income (₹)", min_value=50000.0, max_value=10000000.0, value=500000.0)
    family_members = st.number_input("Number of Family Members", min_value=1, max_value=20, value=3)
    chronic = st.radio("Chronic Diseases?", ['Yes', 'No'])
    flyer = st.radio("Frequent Flyer?", ['Yes', 'No'])
    abroad = st.radio("Ever Travelled Abroad?", ['Yes', 'No'])

    submit = st.form_submit_button("🔍 Predict")

if submit:
    try:
        input_df = pd.DataFrame([{
            'Age': age,
            'Employment Type': 1 if employment_type == 'Private Sector/Self Employed' else 0,
            'GraduateOrNot': 1 if graduate == 'Yes' else 0,
            'AnnualIncome': annual_income,
            'FamilyMembers': family_members,
            'ChronicDiseases': 1 if chronic == 'Yes' else 0,
            'FrequentFlyer': 1 if flyer == 'Yes' else 0,
            'EverTravelledAbroad': 1 if abroad == 'Yes' else 0
        }])

        scaled = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0]

        decision = "✅ You SHOULD get travel insurance." if prediction == 1 else "ℹ️ You MAY NOT need travel insurance."
        confidence = max(probability) * 100

        st.success(decision)
        st.caption(f"Confidence: {confidence:.1f}%")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
