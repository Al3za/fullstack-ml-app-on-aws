
import json
from pathlib import Path

import pandas as pd

from inference import (
    model_fn,
    input_fn,
    predict_fn,
    output_fn,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Questo carica la pipeline e la soglia ottimizzata 0.35.
MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts"

model_artifacts = model_fn(str(MODEL_DIR))

print("Model loaded successfully.")
print("Classification threshold:", model_artifacts["threshold"])

# Read a real customer from the dataset
df = pd.read_csv(DATASET_PATH)

customer = df.drop(
    columns=["customerID", "Churn"]
).iloc[0].to_dict()

# Match the training preprocessing
customer["TotalCharges"] = pd.to_numeric(
    customer["TotalCharges"],
    errors="coerce",
)

if pd.isna(customer["TotalCharges"]):
    customer["TotalCharges"] = 0.0

# Simulate an HTTP JSON request
request_body = json.dumps(customer)

input_data = input_fn(
    request_body,
    "application/json",
)

print("Input shape:", input_data.shape)
print("Input columns:", input_data.columns.tolist())
print("Input validation: OK")


# Generate churn prediction
prediction = predict_fn(
    input_data,
    model_artifacts,
)

print("\nPrediction generated successfully.")
print("Churn probability:", prediction["churn_probability"])
print("Churn prediction:", prediction["churn_prediction"])

# Serialize prediction as an HTTP JSON response
response_body = output_fn(
    prediction,
    "application/json",
)

print("\nJSON response:")
print(response_body)

# Validate the JSON response
response = json.loads(response_body)

assert 0.0 <= response["churn_probability"] <= 1.0
assert response["churn_prediction"] in (0, 1)
assert response["threshold"] == 0.35

print("\nEnd-to-end local inference test: PASSED")
