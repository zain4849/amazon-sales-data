import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras

# Loading saved NN model and necessary components
model = keras.models.load_model("demand_prediction_nn.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("category_encoder.pkl")

st.title("📊 Predict Amazon Product Demand")

st.write("""
### 🚀 Enter Product Details Below to Predict Demand Level
This model predicts whether a product will have **High or Low Demand** based on pricing, discounts, and reviews.
""")

# Category Dropdown
category_options = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Beauty", "Toys", "Sports", "Other"]
category = st.selectbox("Select Product Category:", category_options)

# Convert category to numerical encoding
if category in encoder.classes_:
    category_encoded = encoder.transform([category])[0]
else:
    category_encoded = 0  # Assigning 0 if category is not recognized

# ✅ Discount Percentage Dropdown (Common discount levels)
discount_options = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
discount_percentage = st.selectbox("Select Discount Percentage (%):", discount_options)

# ✅ Number Inputs for Other Features
col1, col2 = st.columns(2)

with col1:
    discounted_price = st.number_input("Discounted Price (₹)", min_value=0.0, value=500.0)
    actual_price = st.number_input("Actual Price (₹)", min_value=0.0, value=1000.0)

with col2:
    rating = st.number_input("Product Rating (1-5)", min_value=1.0, max_value=5.0, value=4.0)
    rating_count = st.number_input("Number of Reviews", min_value=0, value=1000)

# Price difference
price_difference = actual_price - discounted_price

# Predict Demand when button is clicked
if st.button("Predict Demand"):
    input_data = np.array([[discount_percentage, rating, rating_count, category_encoded, price_difference]])
    
    input_scaled = scaler.transform(input_data)
    
    predicted_prob = model.predict(input_scaled)[0][0]
    
    # Converting prediction to High or Low demand
    predicted_demand = "High" if predicted_prob >= 0.5 else "Low"
    
    st.success(f"📌 Predicted Demand Level: **{predicted_demand}**")
    st.write(f"🧠 Model Confidence: **{predicted_prob:.2%}**")

    st.write("""
    **🔹 Interpretation:**  
    - If demand is **High**, the product is likely to be **popular and sell well**.  
    - If demand is **Low**, it may not attract enough buyers. Consider **adjusting price, discounts, or marketing strategies**.
    """)

# Model Performance Summary
st.write("### 📈 Model Performance Metrics:")
st.write("""
✅ **Accuracy:** 98.98%  
✅ **Precision:** 98.00%  
✅ **Recall:** 100.00%  
✅ **F1-Score:** 98.99%  
✅ **ROC-AUC Score:** 99.97%  
""")

st.write("⚡ **This model was trained using a Neural Network to optimize demand prediction!**")

# Footer
st.write("🚀 **Built with Streamlit & TensorFlow** by Zain Ali  🎯")
