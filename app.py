import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

#load trained models
with open('model.pkl', 'rb') as f:
    risk_model = pickle.load(f)

with open('reward_model.pkl', 'rb') as f:
    reward_model = pickle.load(f)

#app title and description
st.title("Risk-Reward Simulator Using Daily Return and 7-Day Moving Average")
st.write("""
This app allows you to predict **Risk** and **Reward** categories for stock data using **Daily Return** and **7-Day Moving Average**:
1. **Interactive Predictions**: Enter stock data to predict risk and reward.
2. **Historical Data Analysis**: Upload a CSV file to analyze risk-reward trends over time.
""")

#tabs for individual prediction and historical data analysis
tab1, tab2 = st.tabs(["Interactive Predictions", "Historical Data Analysis"])

#tab 1(predictions)
with tab1:
    st.header("Interactive Predictions")
    daily_return = st.number_input("Enter Daily Return (%):", step=0.01, format="%.2f")
    seven_day_ma = st.number_input("Enter 7-Day Moving Average (%):", step=0.01, format="%.2f")

    if st.button("Predict"):
        try:
            #input
            sample_input = pd.DataFrame([[daily_return, seven_day_ma]], columns=['Daily Return', '7-Day MA'])

            #predictions
            risk_prediction = risk_model.predict(sample_input)[0]
            reward_prediction = reward_model.predict(sample_input)[0]

            #display predictions
            st.subheader(f"Predicted Risk Category: {risk_prediction}")
            st.write("""
            **Risk Category Descriptions**:
            - **Low Risk**: Minimal price fluctuations, safe investments.
            - **Medium Risk**: Moderate price fluctuations, balanced risk and return.
            - **High Risk**: Significant price fluctuations, high potential for losses or gains.
            """)

            st.subheader(f"Predicted Reward Category: {reward_prediction}")
            st.write("""
            **Reward Category Descriptions**:
            - **Low Reward**: Small expected returns, stable performance.
            - **Medium Reward**: Moderate returns, some risk involved.
            - **High Reward**: High potential returns, with accompanying risks.
            """)

            #visualizing predictions
            st.header("Visualization of Predictions")
            categories = ['Risk', 'Reward']
            values = [1 if risk_prediction == 'High Risk' else (0.5 if risk_prediction == 'Medium Risk' else 0),
                      1 if reward_prediction == 'High Reward' else (0.5 if reward_prediction == 'Medium Reward' else 0)]

            fig, ax = plt.subplots()
            ax.bar(categories, values, color=['blue', 'green'])
            ax.set_ylim(0, 1.1)
            ax.set_ylabel('Category Level')
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(['Low', 'Medium', 'High'])
            ax.set_title('Predicted Risk and Reward Levels')

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

# Tab 2 (historical data analysis)
with tab2:
    st.header("Historical Data Analysis")
    uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

    if uploaded_file is not None:
        #load uploaded file
        historical_data = pd.read_csv(uploaded_file)

        #display uploaded file
        st.write("Uploaded Historical Data:")
        st.dataframe(historical_data.head())

        try:
            #predict
            historical_data['Risk Prediction'] = risk_model.predict(historical_data[['Daily Return', '7-Day MA']])
            historical_data['Reward Prediction'] = reward_model.predict(historical_data[['Daily Return', '7-Day MA']])

            #display predictions
            st.write("Historical Data with Predictions:")
            st.dataframe(historical_data)

            #visualizing trends
            st.header("Visualizations")
            st.line_chart(historical_data[['Daily Return', '7-Day MA']])
            
            st.bar_chart(historical_data['Risk Prediction'].value_counts())
            st.bar_chart(historical_data['Reward Prediction'].value_counts())

            #summary
            avg_daily_return = historical_data['Daily Return'].mean()
            st.metric("Average Daily Return", f"{avg_daily_return:.2f}%")

            st.write("Risk Distribution:")
            st.bar_chart(historical_data['Risk Prediction'].value_counts())

            st.write("Reward Distribution:")
            st.bar_chart(historical_data['Reward Prediction'].value_counts())

        except Exception as e:
            st.error(f"Error during analysis: {e}")
