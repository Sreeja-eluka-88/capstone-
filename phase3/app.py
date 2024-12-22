import streamlit as st
import joblib  # For loading the saved model
import numpy as np
import pandas as pd

# Load the saved XGBoost model from a joblib file
model_path = "xgb_best_model.joblib"  # Update with the path to your saved joblib file
xgb_model = joblib.load(model_path)

# Class mapping for model compatibility
class_mapping = {1: 0, 2: 1, 3: 2, 4: 3}  # Original -> Model mapping
inverse_mapping = {v: k for k, v in class_mapping.items()}  # Model -> Original mapping

# Feature labels
features = ['Start_Lat', 'Start_Lng', 'Distance(mi)', 'Temperature(F)',
            'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Month', 'Hour',
            'Weather_Condition_Label']

# Weather condition options for dropdown
weather_conditions = {'Cloudy': 0, 'Fair': 1, 'Other': 2, 'Rain': 3, 'Smoke': 4, 'Snow': 5, 'Thunder': 6, 'Windy': 7}

# Streamlit app
st.title("Road Severity Prediction")
st.markdown("Enter the following details to predict the severity of road conditions:")

# Input fields for features
start_lat = st.number_input("Start Latitude (Start_Lat)", value=39.0, step=0.1)
start_lng = st.number_input("Start Longitude (Start_Lng)", value=-77.0, step=0.1)
distance = st.number_input("Distance (in miles)", value=1.0, step=0.1)
temperature = st.number_input("Temperature (in °F)", value=70.0, step=1.0)
humidity = st.number_input("Humidity (in %)", value=50.0, step=1.0)
pressure = st.number_input("Pressure (in inches)", value=29.9, step=0.1)
visibility = st.number_input("Visibility (in miles)", value=10.0, step=0.1)
month = st.slider("Month", min_value=1, max_value=12, value=6)
hour = st.slider("Hour (24-hour format)", min_value=0, max_value=23, value=12)

# Dropdown for Weather Condition
weather_condition = st.selectbox("Weather Condition", list(weather_conditions.keys()))
weather_condition_label = weather_conditions[weather_condition]

# Button to predict
if st.button("Predict"):
    # Prepare input features
    input_data = np.array([[start_lat, start_lng, distance, temperature,
                            humidity, pressure, visibility, month, hour,
                            weather_condition_label]])

    # Predict using the XGBoost model
    model_prediction = xgb_model.predict(input_data)
    original_class = inverse_mapping[model_prediction[0]]  # Map back to original class

    # Display the result
    st.success(f"The predicted severity level is: **{original_class}**")
