import pandas as pd
from pathlib import Path

# Percorso assoluto della root del progetto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Percorso del dataset (in locale, non conpatibile su sagemaker)
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


df_analysis = pd.read_csv(DATA_PATH)


# converti i missing value in nan
df_analysis["TotalCharges"] = pd.to_numeric(
    df_analysis["TotalCharges"],
    errors="coerce" # empty data = "" diventano Nan
)

# trasforma i nan value in 0
df_analysis["TotalCharges"]= (df_analysis["TotalCharges"].fillna(0))


# definisci feature e target

target = "Churn"
id_column = "customerID" # da eliminare perche non porta nessun informazione utile al modello

X = df_analysis.drop(columns=[target,id_column])
y = df_analysis[target].map({ # Encoding del target
    "No":0,
    "Yes":1
})





