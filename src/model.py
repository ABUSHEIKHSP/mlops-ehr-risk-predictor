import importlib

def get_model_class(model_import_str: str, model_type_str: str):
    """
    Dynamically import and return the model class from string paths in config.
    Example:
        model_import_str = "sklearn.ensemble"
        model_type_str = "RandomForestClassifier"
    """
    module = importlib.import_module(name=model_import_str)
    model_class = getattr(module, model_type_str)

    return model_class