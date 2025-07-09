from fastapi import FastAPI
import pandas as pd

from src.app.utils import load_model
from src.app.schema import PatientData

app = FastAPI(title="EHR Risk Predictor API")

model = load_model()

@app.get("/")
def root():
    return {"message": "EHR Risk Predictor API is live!"}

@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0])
    }

