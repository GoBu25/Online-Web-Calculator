import joblib
import streamlit as st
import pandas as pd
import numpy as np

# ✅ Load the trained XGBoost model and Label Encoder
model = joblib.load("xgboost_model_vitamin_d.pkl")
label_encoder = joblib.load("label_encoder.pkl")  # To decode predictions

# ✅ Streamlit Web App
st.title("Vitamin D & Calcium Prediction Calculator")

# ✅ User Inputs
age = st.number_input("Enter Age:", min_value=18, max_value=100, value=25)
sex = st.selectbox("Select Gender:", ["Male", "Female"])
calcium = st.number_input("Enter Calcium Level (mg/dL):", min_value=5.0, max_value=15.0, step=0.1)
season = st.selectbox("Select Season:", ["Winter", "Spring", "Summer", "Autumn"])

# ✅ Convert Categorical Inputs to Numeric
sex_encoded = 0 if sex == "Male" else 1
season_encoded = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}[season]

# ✅ Prepare Input Data
user_input = pd.DataFrame({
    "age": [age],
    "sex": [sex_encoded],
    "calcium": [calcium],
    "season": [season_encoded]
})

# ✅ Make Prediction
prediction = model.predict(user_input)
prediction_label = label_encoder.inverse_transform(prediction)[0]  # Convert numeric label back to category

# ✅ Display Result
st.write("### Predicted Vitamin D Status:", prediction_label)
