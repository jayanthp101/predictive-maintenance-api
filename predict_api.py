from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Load Trained Model and Encoders
model = joblib.load(r"B:\predictive_main_bosch_data\predictive_maintenance_model.pkl")
le_type = joblib.load(r"B:\predictive_main_bosch_data\type_label_encoder.pkl")
le_failure = joblib.load(r"B:\predictive_main_bosch_data\failure_label_encoder.pkl")
scaler = joblib.load(r"B:\predictive_main_bosch_data\feature_scaler.pkl")

# Initialize FastAPI
app = FastAPI(title="Predictive Maintenance API", description="🚀 Predict machine failures using AI!")

# Input Schema
class SensorData(BaseModel):
    type: str  # Machine Type (L/M/H)
    air_temp: float
    process_temp: float
    rot_speed: float
    torque: float
    tool_wear: float

# Prediction Endpoint
@app.post("/predict")
def predict_failure(data: SensorData):
    # Encode Machine Type
    try:
        type_encoded = le_type.transform([data.type])[0]
    except ValueError:
        return {"error": "❌ Invalid Machine Type! Use L, M, or H."}

    # Prepare Data for Model
    input_data = np.array([[type_encoded, data.air_temp, data.process_temp, 
                            data.rot_speed, data.torque, data.tool_wear]])
    
    # Apply Feature Scaling
    input_data_scaled = scaler.transform(input_data)

    # Make Prediction
    prediction = model.predict(input_data_scaled)[0]
    predicted_failure = le_failure.inverse_transform([prediction])[0]

    return {"prediction": predicted_failure}

