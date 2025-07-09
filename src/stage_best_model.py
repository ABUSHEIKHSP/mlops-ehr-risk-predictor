import mlflow
from mlflow.client import MlflowClient

from data import load_config

# Config
config = load_config("config/config.yaml")
mlflow_config = config["mlflow"]

EXPERIMENT_NAME = mlflow_config["experiment_name"]
MODEL_NAME = mlflow_config["model_name"]
ALIAS_NAME = mlflow_config["alias_name"]
METRIC = mlflow_config["metric"]
TRACKING_URI = mlflow_config["tracking_uri"]

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"Experiment: {EXPERIMENT_NAME} not found!")

experiment_id = experiment.experiment_id

runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=[f"metric.{METRIC} DESC"],
    max_results=1
)

if not runs:
    raise ValueError("No runs found in this experiment!")

best_run = runs[0]
run_id = best_run.info.run_id
model_uri = f"runs:/{run_id}/model"

registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
print(f"✅ Registered model version: {registered_model.version}")

client.delete_registered_model_alias(name=MODEL_NAME, alias=ALIAS_NAME)
client.set_registered_model_alias(name=MODEL_NAME, alias=ALIAS_NAME, version=registered_model.version)

print(f"✅ Registered version {registered_model.version} as alias '{ALIAS_NAME}' for model '{MODEL_NAME}'")

