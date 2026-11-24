
import json
import os

import joblib
import pandas as pd


def model_fn(model_dir): # carica il modello e la soglia
    """Load the trained pipeline and classification threshold."""

    model_path = os.path.join(model_dir, "xgb_final.joblib")
    threshold_path = os.path.join(model_dir, "threshold.json")

    model = joblib.load(model_path)

    with open(threshold_path, "r", encoding="utf-8") as file:
        threshold_data = json.load(file)

    threshold = float(threshold_data["threshold"])

    return {
        "model": model,
        "threshold": threshold,
    }


def input_fn(request_body, request_content_type): # interpreta la richiesta JSON
    """Parse a JSON request containing customer features."""

    if request_content_type != "application/json":
        raise ValueError(
            f"Unsupported content type: {request_content_type}"
        )

    payload = json.loads(request_body)

    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")

    if not payload:
        raise ValueError("Customer features cannot be empty.")

    return pd.DataFrame([payload])


def predict_fn(input_data, model_artifacts): # calcola la probabilità di churn.
    """Predict churn probability and apply the optimized threshold."""

    model = model_artifacts["model"]
    threshold = model_artifacts["threshold"]

    probabilities = model.predict_proba(input_data)[:, 1]
    probability = float(probabilities[0])

    prediction = int(probability >= threshold)

    return {
        "churn_probability": probability,
        "churn_prediction": prediction,
        "threshold": threshold,
    }


def output_fn(prediction, accept): # restituisce la risposta JSON al client
    """Serialize the prediction as JSON."""

    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")

    return json.dumps(prediction)
