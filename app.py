import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Load the trained XGBoost model with error handling
try:
    model = joblib.load('xgboost_model_vitamin_d_final.pkl')
except Exception as e:
    st.error(f"🚨 Model loading failed: {str(e)}. Ensure the model file is available.")
    st.stop()

# Define the exact feature names expected by the model
expected_features = ['age', 'sex', 'calcium', 'season']

# Streamlit UI
st.title('Vitamin D Classification')

# Input fields for expected features
st.header("Patient Input Parameters")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Age (years):', min_value=18, max_value=100, value=25)
    sex = st.selectbox('Sex:', ['Male', 'Female'])
    
with col2:
    calcium = st.number_input('Calcium Level (mg/dL):', min_value=5.0, max_value=15.0, step=0.1)
    season = st.selectbox('Season:', ['Winter', 'Spring', 'Summer', 'Autumn'])

# Convert categorical features to numeric format
season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
user_input = pd.DataFrame({
    'age': [age],
    'sex': [1 if sex == 'Male' else 0],  # Encode Male=1, Female=0
    'calcium': [calcium],
    'season': [season_map[season]]
})

# Ensure input matches the model's expected structure
user_input = user_input[expected_features]

# Display user input for debugging (Can be disabled in production)
st.write("Processed Input Data:", user_input)

# Make prediction
if st.button("Predict Vitamin D Status"):
    try:
        prediction = model.predict(user_input)[0]

        # Classify the predicted vitamin D status
        categories = {
            'Deficient': (0, 20),
            'Insufficient': (20.1, 29.9),
            'Sufficient': (30, 100),
            'Toxic': (100.1, 300)
        }

        status = "Unknown"
        for category, (low, high) in categories.items():
            if low <= prediction <= high:
                status = category
                break

        # Display results
        st.subheader(f"Vitamin D Status: {status}")
        st.write(f"Predicted Vitamin D Level: {prediction:.1f} ng/mL")

        # Provide recommendations
        if status == "Deficient":
            st.error("⚠️ Vitamin D Deficient")
            st.write("**Recommendations:**")
            st.write("- High-dose supplementation (50,000 IU/week)")
            st.write("- Increase sun exposure (15-30 mins/day)")
        elif status == "Insufficient":
            st.warning("⚠️ Vitamin D Insufficient")
            st.write("**Recommendations:**")
            st.write("- Moderate supplementation (1000-2000 IU/day)")
            st.write("- Dietary adjustments (fatty fish, fortified foods)")
        elif status == "Sufficient":
            st.success("✅ Vitamin D Sufficient")
            st.write("**Maintenance:**")
            st.write("- Continue 600-800 IU/day")
            st.write("- Regular sunlight exposure")
        else:
            st.error("🚨 Vitamin D Toxicity")
            st.write("**Immediate Actions:**")
            st.write("- Discontinue supplements")
            st.write("- Hydration and medical evaluation")

    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

# ✅ Manually store script content
app_code = """{}""".format(
'''import joblib
import streamlit as st
import pandas as pd
import numpy as np
# Load the trained XGBoost model
model = joblib.load('xgboost_model_vitamin_d_final.pkl')
# Define feature names
expected_features = ['age', 'sex', 'calcium', 'season']
# Streamlit UI
st.title('Vitamin D Classification')
# Input fields
age = st.number_input('Age', min_value=18, max_value=100, value=25)
sex = st.selectbox('Sex', ['Male', 'Female'])
calcium = st.number_input('Calcium Level (mg/dL)', min_value=5.0, max_value=15.0, step=0.1)
season = st.selectbox('Season', ['Winter', 'Spring', 'Summer', 'Autumn'])
# Convert categorical features to numeric format
season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
user_input = pd.DataFrame({
    'age': [age],
    'sex': [1 if sex == 'Male' else 0],
    'calcium': [calcium],
    'season': [season_map[season]]
})
# Ensure input matches the model's expected structure
user_input = user_input[expected_features]
# Debugging: Show user input
st.write("User Input Data:", user_input)
# Make prediction
prediction = model.predict(user_input)
# Display prediction result
st.write('Vitamin D Status:', prediction[0])
''')

# ✅ Write script to file
with open("app.py", "w") as file:
    file.write(app_code)

st.success("✅ app.py has been saved successfully!")
