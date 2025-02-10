import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained Gradient Boosting model and scaler
model = joblib.load("rating_gbr_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("⭐ Predict Amazon Product Rating")
st.write("Enter product details below to predict its rating.")

# Input fields
col1, col2 = st.columns(2)
with col1:
    discounted_price = st.number_input("Discounted Price (₹)", min_value=0.0, value=500.0)
    actual_price = st.number_input("Actual Price (₹)", min_value=0.0, value=1000.0)
with col2:
    discount_percentage = st.number_input("Discount Percentage (%)", min_value=0.0, max_value=100.0, value=50.0)
    rating_count = st.number_input("Number of Reviews", min_value=0, value=1000)

# Make Prediction
if st.button("Predict Rating"):
    input_data = np.array([[discounted_price, actual_price, discount_percentage, rating_count]])
    input_data = scaler.transform(input_data)  # Apply the same scaling used during training
    predicted_rating = model.predict(input_data)[0]
    
    # Ensure rating is between 1 and 5
    predicted_rating = np.clip(predicted_rating, 1, 5)

    st.success(f"Predicted Rating: {predicted_rating:.2f}")



# # Input fields
# col1, col2 = st.columns(2)
# with col1:
#     discounted_price = st.number_input("Discounted Price (₹)", min_value=0.0, value=500.0)
#     actual_price = st.number_input("Actual Price (₹)", min_value=0.0, value=1000.0)
# with col2:
#     discount_percentage = st.number_input("Discount Percentage (%)", min_value=0.0, max_value=100.0, value=50.0)
#     rating_count = st.number_input("Number of Reviews", min_value=0, value=1000)

# # Make Prediction
# if st.button("Predict Rating"):
#     input_data = np.array([[discounted_price, actual_price, discount_percentage, rating_count]])
#     input_data = scaler.transform(input_data)
#     predicted_rating = model.predict(input_data)[0][0]
#     if predicted_rating>5:
#        predicted_rating=5
#     st.success(f"Predicted Rating: {predicted_rating:.2f}")