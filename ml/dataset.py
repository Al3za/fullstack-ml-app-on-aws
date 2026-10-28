import pandas as pd

# Carica dataset

DATA_PATH = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df_analysis = pd.read_csv(DATA_PATH)


# converti i missing value in nan
df_analysis["TotalCharges"] = pd.to_numeric(
    df_analysis["TotalCharges"],
    errors="coerce" # i dati = "" diventano Nan
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


# definisci le features

# === FEATURE GROUPS ===

numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

binary_features = [
    "SeniorCitizen"
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

# -----

