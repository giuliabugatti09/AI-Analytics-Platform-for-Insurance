import pandas as pd
import numpy as np
import requests
import joblib
import io

def load_sklearn_model(file):
    bytes_data = file.read()
    try:
        model = joblib.load(io.BytesIO(bytes_data))
    except Exception:
        import pickle
        model = pickle.loads(bytes_data)
    return model

def predict_with_sklearn(model, X: pd.DataFrame, use_proba=True):
    if use_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        else:
            return pd.DataFrame(proba, columns=[f"proba_{i}" for i in range(proba.shape[1])])
    elif hasattr(model, "predict"):
        return model.predict(X)

def predict_with_plumber_api(api_url: str, X: pd.DataFrame):
    payload = {"data": X.to_dict(orient="records")}
    r = requests.post(api_url, json=payload, timeout=30)
    return r.json()

def try_rpy2_predict(rds_file, X: pd.DataFrame, r_fun="predict"):
    raise NotImplementedError("Requer rpy2 e R instalados")
