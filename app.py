import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# -------------------------
# Gauge Function
# -------------------------
def create_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Churn Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "yellow"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))
    return fig


# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title='Customer Churn Predictor', layout='wide')
st.title('📊 Customer Churn Prediction System')


# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_churn_model.pkl")


model_data = load_model()
model = model_data["model"]
model_columns = model_data["columns"]

st.success("✅ Model loaded successfully!")


# -------------------------
# Inputs
# -------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])

with col2:
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    monthly_charges = st.number_input('Monthly Charges', 0.0, 200.0, 70.0)


# -------------------------
# Prediction
# -------------------------
if st.button('Predict Churn'):

    # input data
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges
    }

    input_df = pd.DataFrame([input_data])

    # one-hot encoding
    input_encoded = pd.get_dummies(input_df)

    # match training columns
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[model_columns]

    # prediction
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0]
    churn_prob = probability[1] * 100


    # -------------------------
    # OUTPUT
    # -------------------------
    st.subheader("📊 Risk Analysis Dashboard")

    st.metric("Churn Probability", f"{churn_prob:.1f}%")

    if prediction == 1:
        st.error("⚠️ High Risk Customer")

        st.warning("📌 Business Action Recommendations")
        st.write("""
        - Offer discount or loyalty bonus  
        - Contact customer for feedback  
        - Provide priority support  
        - Run retention campaigns  
        """)

    else:
        st.success("✅ Low Risk Customer")

        st.info("📌 Growth Recommendations")
        st.write("""
        - Upsell premium services  
        - Offer referral rewards  
        - Improve engagement programs  
        - Build long-term loyalty  
        """)

    # Gauge chart
    st.plotly_chart(create_gauge(churn_prob / 100), use_container_width=True)
