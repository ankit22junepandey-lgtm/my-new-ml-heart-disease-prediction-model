from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved model
model = joblib.load("model.pkl")

# Create FastAPI app
app = FastAPI(title="Heart Disease Prediction API")

# HTML templates
templates = Jinja2Templates(directory="templates")

# Input schema for JSON API
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

# Function to predict
def predict_heart_disease(data):
    features = np.array([list(data.values())]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return "🧡 No Heart Disease Detected" if prediction == 0 else "❤️ High Risk of Heart Disease"

# Home route (frontend form)
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Prediction route (from form)
@app.post("/", response_class=HTMLResponse)
def predict_from_form(
    request: Request,
    age: int = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: int = Form(...),
    chol: int = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: int = Form(...),
    exang: int = Form(...),
    oldpeak: float = Form(...),
    slope: int = Form(...),
    ca: int = Form(...),
    thal: int = Form(...)
):
    data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    result = predict_heart_disease(data)
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

# JSON API (for external apps or Postman)
@app.post("/predict/")
def predict_api(data: HeartData):
    result = predict_heart_disease(data.dict())
    return {"prediction": result}
