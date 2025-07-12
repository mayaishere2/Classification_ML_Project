import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('MLmodel_LinearSVC.pkl')

# App title and description
st.title("🏆 Predicting Winning Teams in Dota 2 Matches")

st.write(
    """
    This machine learning app predicts whether a Dota 2 team wins a match based on in-game statistics.
    for more information about the dataset, and analysis on the problem and how we chose the model, please refer to the notebook:
    https://colab.research.google.com/drive/1lF4ouQxq8ZNYSvH51SRTshRQJKPUgvxC
    - The data format must match the Kaggle dataset:
      [Dota 2 Matches Dataset](https://www.kaggle.com/datasets/ashishpatel26/dota-2-matches)
    - You can use the example dataset below to test the model.
    """
)
# Example data (to be previewed)
data_cleaned = pd.read_csv('data_cleaned.csv')  # lowercase 'cleaned'

# Display a preview
st.subheader("📄 Example Data (data_cleaned.csv)")
st.dataframe(data_cleaned.head())

# Download link for users to get the example CSV
st.download_button(
    label="📥 Download Example CSV",
    data=data_cleaned.to_csv(index=False),
    file_name="example_dota2_data.csv",
    mime="text/csv"
)

# File uploader for user to upload their own test data
st.subheader("🧪 Upload Your Match Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read the CSV file
        user_data = pd.read_csv(uploaded_file)
        st.write("✅ File uploaded successfully. Here’s a preview:")
        st.dataframe(user_data.head())

        # Prediction
        predictions = model.predict(user_data)
        st.subheader("🔮 Predictions:")
        st.write(predictions)

    except Exception as e:
        st.error(f"⚠️ Error reading or predicting on uploaded file: {e}")
