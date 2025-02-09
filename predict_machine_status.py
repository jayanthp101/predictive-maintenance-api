import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 🔹 Step 1: Load the Trained AI Model & Label Encoder
model = joblib.load(r"B:\predictive_main_bosch_data\predictive_maintenance_model.pkl")
le_failure = joblib.load(r"B:\predictive_main_bosch_data\failure_label_encoder.pkl")  # Load encoder used during training

# 🔹 Step 2: Take New Sensor Readings
type_input = input("Enter Machine Type (L/M/H): ").strip().upper()
air_temp = float(input("Enter Air Temperature (K): "))
process_temp = float(input("Enter Process Temperature (K): "))
rot_speed = float(input("Enter Rotational Speed (rpm): "))
torque = float(input("Enter Torque (Nm): "))
tool_wear = float(input("Enter Tool Wear (min): "))

# 🔹 Step 3: Encode Type (Ensure consistency with training)
type_map = {'L': 0, 'M': 1, 'H': 2}  # Adjust based on LabelEncoder mapping
type_encoded = type_map.get(type_input, -1)  # Default to -1 if invalid input

if type_encoded == -1:
    print("❌ Invalid Machine Type! Please enter L, M, or H.")
    exit()

# 🔹 Step 4: Make a Prediction (Fixing Feature Name Issue)
feature_names = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
new_data = pd.DataFrame([[type_encoded, air_temp, process_temp, rot_speed, torque, tool_wear]], columns=feature_names)

prediction = model.predict(new_data)[0]

# 🔹 Step 5: Display Prediction
predicted_failure = le_failure.inverse_transform([prediction])[0]  # Decode failure type

if predicted_failure == "No Failure":
    print("✅ Machine is RUNNING normally.")
else:
    print(f"⚠️ Machine is likely to FAIL due to: {predicted_failure}. Perform Maintenance!")
