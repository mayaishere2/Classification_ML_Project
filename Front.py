import streamlit as st
import joblib
import pandas as pd 

model = joblib.load('MLmodel_LinearSVC.pkl')

st.title("Predicting winning teams in Dota 2 Matches")

uploaded_file = st.file_uploader("Choose a CSV file with match data", type="csv")

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    data = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the DataFrame
    st.write("Data Preview:")
    st.dataframe(data.head())
    predictions = model.predict(data)
    # Display the predictions
    st.write("Predictions:")
    st.write(predictions)