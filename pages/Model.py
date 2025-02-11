import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras

# Load trained Neural Network model and necessary components
model = keras.models.load_model("demand_prediction_nn.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("category_encoder.pkl")

# Streamlit UI
st.title("📊 Predict Amazon Product Demand")

st.write("""
### 🚀 Enter Product Details Below to Predict Demand Level
This model predicts whether a product will have **High or Low Demand** based on pricing, discounts, and reviews.
""")

# ✅ Input fields for user
col1, col2 = st.columns(2)

with col1:
    discounted_price = st.number_input("Discounted Price (₹)", min_value=0.0, value=500.0)
    actual_price = st.number_input("Actual Price (₹)", min_value=0.0, value=1000.0)
    discount_percentage = st.number_input("Discount Percentage (%)", min_value=0.0, max_value=100.0, value=50.0)

with col2:
    rating = st.number_input("Product Rating (1-5)", min_value=1.0, max_value=5.0, value=4.0)
    rating_count = st.number_input("Number of Reviews", min_value=0, value=1000)
    category = st.text_input("Product Category (e.g., Electronics, Clothing, etc.)", value="Electronics")

# Convert category to numerical encoding
if category in encoder.classes_:
    category_encoded = encoder.transform([category])[0]
else:
    category_encoded = 0  # Assigning 0 if category is not recognized

# Calculate price difference
price_difference = actual_price - discounted_price

# ✅ Predict Demand when button is clicked
if st.button("Predict Demand"):
    # Create input array
    input_data = np.array([[discount_percentage, rating, rating_count, category_encoded, price_difference]])
    
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    predicted_prob = model.predict(input_scaled)[0][0]
    
    # Convert prediction to High or Low demand
    predicted_demand = "High" if predicted_prob >= 0.5 else "Low"
    
    # ✅ Display the result
    st.success(f"📌 Predicted Demand Level: **{predicted_demand}**")
    st.write(f"🧠 Model Confidence: **{predicted_prob:.2%}**")

    # ✅ Display explanation
    st.write("""
    **🔹 Interpretation:**  
    - If demand is **High**, the product is likely to be **popular and sell well**.  
    - If demand is **Low**, it may not attract enough buyers. Consider **adjusting price, discounts, or marketing strategies**.
    """)

# 📊 Model Performance Summary
st.write("### 📈 Model Performance Metrics:")
st.write("""
✅ **Accuracy:** 98.98%  
✅ **Precision:** 98.00%  
✅ **Recall:** 100.00%  
✅ **F1-Score:** 98.99%  
✅ **ROC-AUC Score:** 99.97%  
""")

st.write("⚡ **This model was trained using a Neural Network to optimize demand prediction!**")

# ✅ Footer
st.write("🚀 **Built with Streamlit & TensorFlow** | Data Science by [Your Name] 🎯")
