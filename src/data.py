import yaml
import pandas as pd

# Load yaml config
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config

# Load data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df