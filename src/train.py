import argparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from data import load_config, load_data
from model import get_model_class
from pipeline import get_column_types, get_preprocessor, build_pipeline

# This file has to be run from root/main dir and not from the src dir

# Initilaizing parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/config.yaml")
parser.add_argument("--run_name", default=None)
args = parser.parse_args()

# Load config
config = load_config(args.config)

# Load dataset
data = load_data(config["data"]["path"])

# Features and Target
X = data.drop(config["data"]["target_column"], axis=1)
y = data[config["data"]["target_column"]]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"])

# Get model class
model_cls = get_model_class(model_import_str=config["model"]["model_module"], model_type_str=config["model"]["model_type"])

# Preprocessing pipeline
cat_cols, num_cols = get_column_types(X_train=X_train)
preprocessor = get_preprocessor(cat_cols, num_cols)
pipeline = build_pipeline(preprocessor=preprocessor, model_cls=model_cls, model_params=config["model"]["model_params"])

# Start Mlflow experiment
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

with mlflow.start_run(run_name=args.run_name):
    # log model params & data config
    mlflow.log_param("model_type", config["model"]["model_type"])
    mlflow.log_params(config["model"]["model_params"])
    mlflow.log_params({
        "test_size": config["data"]["test_size"],
        "random_state": config["data"]["random_state"]
    })

    # train
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    # metrics
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("fl_score", f1)

    # log model
    signature = infer_signature(X_test, pipeline.predict(X_test))
    mlflow.sklearn.log_model(
        pipeline, 
        name="model", 
        signature=signature, 
        input_example=X_test.iloc[:2]
    )

    print(f"✅ Logged metrics - Accuracy: {acc}, F1 Score: {f1}")

