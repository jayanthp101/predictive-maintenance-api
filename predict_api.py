from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os
from pydantic import BaseModel

# Define paths for model and encoders
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "predictive_maintenance_model.pkl")
type_encoder_path = os.path.join(BASE_DIR, "type_label_encoder.pkl")
failure_encoder_path = os.path.join(BASE_DIR, "failure_label_encoder.pkl")
scaler_path = os.path.join(BASE_DIR, "feature_scaler.pkl")

# Load Model and Encoders (Handle missing files)
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        raise FileNotFoundError(f"❌ Missing file: {path}")

model = load_model(model_path)
le_type = load_model(type_encoder_path)
le_failure = load_model(failure_encoder_path)
scaler = load_model(scaler_path)

# Initialize FastAPI
app = FastAPI(title="Predictive Maintenance API", description="🚀 Predict machine failures using AI!")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

