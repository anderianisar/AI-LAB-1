# =============================
# Gauge Chart
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
        ],
    }
))

st.plotly_chart(fig, use_container_width=True)
