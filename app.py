import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title='Customer Churn Predictor',
    page_icon='📊',
    layout='wide'
)

st.title('📊 Customer Churn Prediction System')

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    with open('best_churn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.success("Model loaded successfully!")

# =============================
# INPUT FORM
# =============================
col1, col2 = st.columns(2)

with col1:
    st.subheader('Customer Demographics')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['No', 'Yes'])
    dependents = st.selectbox('Dependents', ['No', 'Yes'])

with col2:
    st.subheader('Account Information')
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    monthly_charges = st.number_input(
        'Monthly Charges ($)',
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

# =============================
# PREDICTION BUTTON
# =============================
if st.button('Predict Churn', type='primary'):

    # Input data
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges
    }

    input_df = pd.DataFrame([input_data])

    # Encoding
    input_encoded = pd.get_dummies(input_df)

    # Align with model
    input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    # Prediction
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0]
    churn_prob = probability[1] * 100

    # =============================
    # RESULT
    # =============================
    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.error("⚠️ HIGH RISK: Customer likely to churn")
        st.metric("Churn Probability", f"{churn_prob:.1f}%")
    else:
        st.success("✅ LOW RISK: Customer likely to stay")
        st.metric("Retention Probability", f"{100 - churn_prob:.1f}%")

    # =============================
    # GAUGE CHART
    # =============================
    st.subheader("📊 Churn Probability Gauge")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob,
        title={'text': "Churn Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # =============================
    # BUSINESS RECOMMENDATIONS
    # =============================
    st.subheader("💡 Business Recommendations")

    if prediction == 1:
        st.warning("High-risk customer detected!")

        st.info("👉 Offer discount or special retention plan.")
        st.info("👉 Improve customer support follow-ups.")
        st.info("👉 Provide loyalty rewards.")
        st.info("👉 Investigate dissatisfaction reasons.")
    else:
        st.success("Low-risk customer")

        st.info("👉 Upsell premium services.")
        st.info("👉 Offer loyalty programs.")
        st.info("👉 Encourage referrals.")
