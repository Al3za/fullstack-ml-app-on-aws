# # La responsabilità di questo file sarà:

# # 1. Ricevere i dati di un cliente.
# # 2. Applicare la pipeline salvata (xgb_final.joblib), che include già il preprocessing.
# # 3. Calcolare la probabilità di churn.
# # 4. Applicare la soglia 0.35.
# # 5. Restituire un risultato strutturato.
# #    Il preprocessing rimane nella pipeline: non dobbiamo riscrivere l'One-Hot Encoding in Lambda.


# # Realizziamo una funzione Python indipendente da AWS, così possiamo verificarla localmente prima di introdurre il formato richiesto dal container SageMaker.

import json
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgb_final.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"


def load_artifacts():
    """Carica la pipeline addestrata e la soglia decisionale."""
    model = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold_config = json.load(file)

    threshold = float(threshold_config["threshold"])

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("La soglia deve essere compresa tra 0 e 1.")

    return model, threshold


def predict_customer(customer_data):
    """
    Riceve un dizionario con le feature di un cliente
    e restituisce probabilità e classe predetta.
    """
    model, threshold = load_artifacts()

    # La pipeline si aspetta le colonne originali come in training.
    input_df = pd.DataFrame([customer_data])

    probabilities = model.predict_proba(input_df)[:, 1]
    churn_probability = float(probabilities[0])

    prediction = int(churn_probability >= threshold)

    return {
        "churn_probability": churn_probability,
        "churn_prediction": prediction,
        "threshold": threshold,
    }


# Importa questo file sul file xgboost