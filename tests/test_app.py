import warnings
import pytest
from fastapi.testclient import TestClient

from src.app.predict import app

client = TestClient(app)

# Sample valid data
valid_payload = {
    "Age": 54,
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingBP": 140,
    "Cholesterol": 239,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 160,
    "ExerciseAngina": "N",
    "Oldpeak": 1.2,
    "ST_Slope": "Up"
}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "EHR Risk Predictor API is live!"

def test_prediction_endpoint():
    response = client.post(url="/predict", json=valid_payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)


# To test, call from project root dir,
# PYTHONPATH=. pytest -p no:warnings
