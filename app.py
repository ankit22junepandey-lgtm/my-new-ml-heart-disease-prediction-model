from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load saved model
model = joblib.load("model.pkl")

# Create FastAPI app
app = FastAPI(title="Heart Disease Prediction API")

# Input schema
class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Prediction function
def predict_heart_disease(data):
    features = np.array([list(data.values())]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return "🧡 No Heart Disease Detected" if prediction == 0 else "❤️ High Risk of Heart Disease"

# Root route
@app.get("/")
def home():
    return {"message": "Welcome to the Heart Disease Prediction API!"}

# Prediction route
@app.post("/predict/")
def predict(data: HeartData):
    result = predict_heart_disease(data.dict())
    return {"prediction": result}
