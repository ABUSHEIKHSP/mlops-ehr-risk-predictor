import mlflow
import joblib
import os
from data import load_config

# Load config
config = load_config("config/config.yaml")
mlflow_config = config["mlflow"]

# Define model alias and path
MODEL_NAME = mlflow_config["model_name"]
ALIAS_NAME = mlflow_config["alias_name"]
SAVE_DIR = "models"
SAVE_PATH = os.path.join(SAVE_DIR, "best_model.pkl")

# Load model from MLflow using alias
model_uri = f"models:/{MODEL_NAME}@{ALIAS_NAME}"
model = mlflow.pyfunc.load_model(model_uri)

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Save the model
joblib.dump(model, SAVE_PATH)

print(f"✅ Model saved to {SAVE_PATH}")
