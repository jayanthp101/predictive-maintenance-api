import streamlit as st
import requests

# API URL
API_URL = "https://predictive-maintenance-api.onrender.com/predict"

st.title("🔧 Predictive Maintenance AI")

# Input fields
type_ = st.selectbox("Machine Type", ["L", "M", "H"])
air_temp = st.number_input("Air Temperature (K)", value=297)
process_temp = st.number_input("Process Temperature (K)", value=308)
rot_speed = st.number_input("Rotational Speed (rpm)", value=2861)
torque = st.number_input("Torque (Nm)", value=4.4)
tool_wear = st.number_input("Tool Wear (min)", value=140)

if st.button("🔍 Predict Failure"):
    data = {
        "type": type_,
        "air_temp": air_temp,
        "process_temp": process_temp,
        "rot_speed": rot_speed,
        "torque": torque,
        "tool_wear": tool_wear,
    }
    
    response = requests.post(API_URL, json=data)
    
    if response.status_code == 200:
        st.success(f"⚠️ Prediction: {response.json()['prediction']}")
    else:
        st.error("❌ Error: Could not get a response from API")

