
# # Inseriamo gli import e la configurazione.

import json
import os

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from preprocessing import make_preprocessor
from feature_config import (
    numeric_features,
    binary_features,
    categorical_features
)

from sagemaker.train.configs import SourceCode


# Configurazione dei percorsi e lettura dei dati

# Con SourceCode inviamo il codice sorgente locale ad AWS, in modo che il container possa eseguire train.py e trovare i moduli necessari cosi' come qui' in locale
SourceCode(
    source_dir="sagemaker/training", # Cartella locale contenente il codice da inviare
    entry_script="train.py", # Script principale da eseguire
    requirements="requirements.txt" # File con le dipendenze Python da installare
)


TRAINING_DIR = os.environ.get(
    "SM_CHANNEL_TRAIN", # SageMaker legge il CSV dalla cartella di input del container, tramite la variabile SM_CHANNEL_TRAIN
    "/opt/ml/input/data/train"# Il file sistem path del container Docker avviato da Sagemaker che contiene i dati di train presi da s3 (il dataset Teleco che abbiamo caricato precedentemente con boto3). 
)

MODEL_DIR = "/opt/ml/model" # Al termine del training, SageMaker raccoglie gli artefatti salvati in questa directory e li carica su S3 per poter fare inference.


# Soglia decisionale selezionata durante il tuning locale
THRESHOLD = 0.35


# Aggiungiamo la funzione di caricamento


# Dataset gia' analizato in locale. Qui si effetuano solo le stesse trasformazioni dati per l'allenamento su sagemaker

def load_training_data():

    train_file = os.path.join(TRAINING_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")  

    df = pd.read_csv(train_file)

    # Conversione di TotalCharges in numerico
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Sostituzione dei valori mancanti
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Separazione target e feature
    target = "Churn"
    id_column = "customerID"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    y = df[target].map({
        "No": 0,
        "Yes": 1
    })

    if y.isna().any():
        raise ValueError("Target contains missing or unexpected values.")

    X = df.drop(columns=[target, id_column], errors="ignore")

    # Verifica delle feature richieste dal preprocessing
    expected_features = (
        numeric_features
        + binary_features
        + categorical_features
    )

    missing_features = set(expected_features) - set(X.columns)

    if missing_features:
        raise ValueError(
            f"Missing expected features: {missing_features}"
        )
    

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y 
    )


    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")

    return X_train, y_train # Usiamo solo i dati di train che ci servono per allenare il modello su Sagemaker

# ----
# Ricostruiamo la pipeline
# ricreare su SageMaker lo stesso modello che hai addestrato localmente

def build_model():

    # Ricreiamo il preprocessing
    preprocessor = make_preprocessor(
        numeric_features=numeric_features,
        binary_features=binary_features,
        categorical_features=categorical_features
    )

    # Costruiamo la pipeline XGBoost
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,

            # Iperparametri selezionati con RandomizedSearchCV
            colsample_bytree=0.8211572356285312,
            gamma=1.185393447758787,
            learning_rate=0.012740679849625696,
            max_depth=4,
            min_child_weight=1,
            n_estimators=354,
            subsample=0.9212559025519583
        ))
    ])

    return model

# ---

# Training e salvataggio del modello


def train_model():

    # Carichiamo i dati di train
    X_train, y_train = load_training_data()

    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")

    # Costruiamo la pipeline
    model = build_model()

    # Addestriamo il modello
    model.fit(X_train, y_train)

    print("Training completed.")

    return model


# ---
# Funzione per salvare gli artefatti 

def save_artifacts(model):

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Salviamo la pipeline completa
    model_path = os.path.join(MODEL_DIR, "xgb_final.joblib")
    joblib.dump(model, model_path)

    # Salviamo la soglia decisionale
    threshold_path = os.path.join(MODEL_DIR, "threshold.json")

    with open(threshold_path, "w", encoding="utf-8") as file:
        json.dump(
            {"threshold": THRESHOLD},
            file,
            indent=2
        )

    print(f"Model saved to: {model_path}")
    print(f"Threshold saved to: {threshold_path}")

# ---

# Punto di ingresso dello script

if __name__ == "__main__":

    model = train_model()

    save_artifacts(model)

    print("SageMaker training script completed successfully.")
