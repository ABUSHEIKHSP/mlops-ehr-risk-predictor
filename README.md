# 🩺 mlops-ehr-risk-predictor

A fully **end-to-end machine learning project** for predicting health risks using Electronic Health Records (EHR).  
This project demonstrates how to build **real-world, production-ready ML pipelines** with **MLOps best practices**, from training to deployment — without overwhelming beginners.

---

## 🚀 Features

- ✅ Config-driven ML pipeline (`.yaml`) — no hardcoded params!
- 🧼 Preprocessing using Scikit-learn Pipelines
- 🤖 Model training, evaluation, and selection
- 📊 MLflow for experiment tracking, model registry, versioning
- 🛠️ REST API using **FastAPI**
- 🐳 Dockerized API for deployment
- 📦 Final model saved locally for serving
- ✅ Optional MLflow-based model loading (via alias)
- 🧪 Unit testing with `pytest`
- 📁 Clean modular folder structure for real-world scaling

---

## 🧱 Project Structure

```plaintext
ehr-risk-predictor-mlops/
│
├── config/
│   └── config.yaml             # All configs (paths, model params, MLflow, etc.)
│
├── data/
│   └── heart.csv              # Raw dataset (you can replace this)
│
├── models/
│   └── model.pkl              # Final saved model (after training + staging)
│
├── src/
│   ├── data.py                # Load & preprocess data
│   ├── train.py               # Main training script
│   ├── stage_best_model.py    # Stage best MLflow model (using alias)
│   ├── save_model.py          # Save staged model locally to models/
│   └── app/
│       ├── predict.py         # FastAPI app
│       ├── schema.py          # Pydantic schema for POST data
│       └── utils.py           # Model loader (local or MLflow)
│
├── tests/
│   └── test_app.py            # Simple unit test for the API
│
├── Dockerfile                 # Containerize the FastAPI + model
├── requirements.txt
├── .dockerignore
└── README.md
```

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/your-username/mlops-ehr-risk-predictor.git
cd ehr-risk-predictor-mlops

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
---

## ⚙️ Configuration

All settings are defined in one place: config/config.yaml

```yaml
data:
  path: data/heart.csv
  target_column: HeartDisease
  test_size: 0.2
  random_state: 42

model:
  model_type: RandomForestClassifier
  model_module: sklearn.ensemble
  model_params:
    n_estimators: 100
    max_depth: 5
    random_state: 42

mlflow:
  experiment_name: ehr-risk-experiment
  tracking_uri: mlruns
  model_name: ehr_model
  alias_name: Champion
  metric: f1
```
---

## 🧠 Train the Model

Train your own model using the current config:

```bash
python src/train.py
```

Automatically logs experiments to MLflow.

---

## 🏷️ Stage & Save Best Model

```bash
# Tag the best model with alias from MLflow
python src/stage_best_model.py

# Save it locally as model.pkl
python src/save_model.py
```
---

## 🚀 Run the API

```bash
uvicorn src.app.predict:app --host 0.0.0.0 --port 8000
```
---

## Test with curl:

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
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
         }'
```
---

## 🧪 Run Tests

```bash
PYTHONPATH=. pytest
```
---

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t ehr-risk-api .

# Run container
docker run -p 8000:8000 ehr-risk-api
```
---

## 📌 How to Use This Project

This project is meant to be beginner-friendly and modular.

- You don’t need to modify any code to train a model — just edit `config/config.yaml`.
- Run `train.py` to train and track your model with MLflow.
- Run `stage_best_model.py` to tag the best model in MLflow.
- Run `save_model.py` to export the model locally.
- The FastAPI app (`predict.py`) uses this local model to serve predictions.
- You can test the API with `curl` or integrate it with other systems.
- The app is fully dockerized for easy deployment.

Start small, one step at a time.

