import streamlit as st
import pandas as pd
import joblib
import numpy as np

model=joblib.load('src/models/xgb_model.pkl')

st.title("TeleCom Customer Churn Prediction App")
st.header("Enter Customer Details")

col1,col2=st.columns(2)

with col1:
    age=st.number_input("Age (in years)", min_value=12, max_value=83, value=45)
    contract_type=st.selectbox("Contract Type", options=["Month-to-Month", "One-Year", "Two-Year"])
    total_charges=st.number_input("Total Charges (in $)", min_value=0.0, max_value=12416.25, value=872.87, format="%.2f")

with col2:
    gender = st.selectbox("Gender", options=["Male", "Female"])
    tenure = st.number_input("Tenure (in months)", min_value=0, max_value=122, value=13)
    monthly_charges = st.number_input("Monthly Charges (in $)", min_value=30.0, max_value=119.96, value=74.06, format="%.2f")

internet_service = st.selectbox("Internet Service", options=["Fiber Optic", "DSL", "Unknown"])
tech_support = st.selectbox("Is tech support required", options=["Yes", "No"])

def prepare_input(tenure, monthly_charges, contract_type, internet_service):
    input_data=pd.DataFrame({
        'Age': [age],
        'Tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges], 
        'Gender_Female': [1 if gender=="Female" else 0],
        'Gender_Male': [1 if gender=="Male" else 0],  
        'ContractType_Month-to-Month': [1 if contract_type == "Month-to-Month" else 0],
        'ContractType_One-Year': [1 if contract_type == "One-Year" else 0],
        'ContractType_Two-Year': [1 if contract_type == "Two-Year" else 0],
        'InternetService_DSL': [1 if internet_service == "DSL" else 0],
        'InternetService_Fiber Optic': [1 if internet_service == "Fiber Optic" else 0],
        'InternetService_Unknown': [1 if internet_service == "Unknown" else 0],
        'TechSupport_No': [1 if tech_support=="No" else 0],
        'TechSupport_Yes': [1 if tech_support=="Yes" else 0]     
    })
    return input_data

if st.button("Predict Churn"):
    input_df=prepare_input(tenure,monthly_charges,contract_type,internet_service)

    prediction=model.predict(input_df)[0]
    prediction_prob=model.predict_proba(input_df)[0]

    churn_result=("Unfortunately, this customer is likely to leave our telecom services." if prediction==1 
                else "Great news! This customer is likely to stay with our telecom network.")
    st.markdown(f"<div style='text-align:center; font-size:32px; font-weight:bold; padding:20px;'>{churn_result}</div>",
                unsafe_allow_html=True)

    st.subheader(f"Probability of Churn:{prediction_prob[1]:.2%}" if(prediction==1) else f"Probability of No Churn:{prediction_prob[0]:.2%}")