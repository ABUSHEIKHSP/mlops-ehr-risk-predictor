import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_column_types(X_train: pd.DataFrame) -> tuple:
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["int", "float"]).columns.tolist()

    return cat_cols, num_cols

def get_preprocessor(cat_cols, num_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat_transformer", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num_transformer", StandardScaler(), num_cols)
        ],
        remainder="passthrough"
    )

    return preprocessor

def build_pipeline(preprocessor: ColumnTransformer, model_cls, model_params: dict):
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", model_cls(**model_params))
        ]
    )

    return pipeline

