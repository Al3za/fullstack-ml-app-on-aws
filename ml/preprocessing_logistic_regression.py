# Logistic Regression

import pandas as pd

DATA_PATH = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH) # i dati raw non modificati

df_analysis = df.copy() # dati modificati

# Transform TotalCharges empty data "" to 0 to perform  on the column:

df_analysis["TotalCharges"] = pd.to_numeric( # Convert argument to a numeric type.
    df_analysis["TotalCharges"],
    errors="coerce" # sets the "" values to NAN
)

df_analysis['TotalCharges'] = df_analysis['TotalCharges'].fillna(0)

print((df_analysis['TotalCharges'] == 0).sum())

print("=== PREPROCESSING - TASK 1: FEATURE PREPARATION ===")

target = "Churn"
id_column = "customerID"

X = df_analysis.drop(columns=[target,id_column])
y = df_analysis[target]

print("\n--- X ---")
print(X.shape)

print("\n--- y ---")
print(y.shape)

print("\n--- Feature utilizzate ---")
print(X.columns.tolist()) # crea un list/array delle colonne di X. Id e' stato tolto correttamente

print("\n--- Target ---")
print(y.name)


# TASK 2 — Pulizia e tipizzazione delle feature
# Partiamo da X, non da df_analysis, perché abbiamo già separato target e identificatore.

print("\n--- Data types prima della conversione ---")
print(X.dtypes)

print("\n--- Missing values prima della conversione ---")
print(X.isna().sum().sort_values(ascending=False))


# ---------
# TASK 3 — Separazione delle feature numeriche e categoriche

print("=== PREPROCESSING - TASK 3: FEATURE GROUPS ===")

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

print("Numeric features:")
print(numeric_features)

print("\nBinary features:")
print(binary_features)

print("\nCategorical features:")
print(categorical_features)

print("\nNumero feature numeriche:", len(numeric_features))
print("Numero feature binarie:", len(binary_features))
print("Numero feature categoriche:", len(categorical_features))


# -------------

# TASK 4 — Encoding del target

print("=== PREPROCESSING - TASK 4: TARGET ENCODING ===")

print("\n--- Target prima della conversione ---")
print(y.value_counts())

y = y.map({
    "No": 0,
    "Yes": 1
})

print("\n--- Target dopo la conversione ---")
print(y.value_counts())

print("\n--- Target dtype ---")
print(y.dtype)

print("\n--- Valori mancanti ---")
print(y.isna().sum())

# ---------

# TASK 5 — Encoding delle feature categoriche

from sklearn.preprocessing import OneHotEncoder

categorical_encoder = OneHotEncoder(
    handle_unknown="ignore", # non blocca l'encoder se si presenta una categoria nuova durente l'inferenza
    sparse_output=False, # sparse_output=True utile con molte categorie
    # drop="first" inseriscilo
)

print(categorical_encoder)


# --------------

# TASK 6 — Costruire il ColumnTransformer

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric", # nome scelti da me per identificare i vari trasformatori.
            "passthrough", # parola riconosciuta da ColumnTransformer. significa "Passa queste colonne senza modificarle" 
            numeric_features
        ),
        (
            "binary",
            "passthrough",
            binary_features
        ),
        (
            "categorical",
            categorical_encoder, # ColumnTransformer trasforma i dati di categorical_features con one-Hot_encoder definito sopra
            categorical_features
        )
    ]
)

print(preprocessor)

print("\n--- Feature totali ---")
print(len(numeric_features) + len(binary_features) + len(categorical_features)) # 19

print("\n--- Preprocessor configurato ---")
print(preprocessor.transformers) ## "Fammi vedere la configurazione dei trasformatori che ho definito."


# --------
# TASK 7 — Train/Test Split e Fit e trasformazione delle feature

from sklearn.model_selection import train_test_split

print("=== PREPROCESSING - TASK 7.1: TRAIN / TEST SPLIT ===")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- Shapes ---")
print("X_train", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\n--- Target distribution TRAIN ---")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True)*100)

print("\n--- Target distribution TEST ---")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True) * 100)


# ----
# Eseguiamo il fit e la trasformazione

print("=== PREPROCESSING - TASK 7.2: FIT & TRANSFORM ===")

X_train_processed = preprocessor.fit_transform(X_train) # one hot encoder applied 
X_test_processed = preprocessor.transform(X_test)


print("\n--- Shapes dopo preprocessing ---")
print("X_train_processed:", X_train_processed.shape)
print("X_test_processed :", X_test_processed.shape)
# 45 colonne delle x features dopo transformazione encoding

# Trasformiamo l'arrey numpy in dataframe pandas solo per vedere la trasformazione dell encoder sulle var categortiche¶
feature_names = preprocessor.get_feature_names_out() # get column

X_train_processed_df = pd.DataFrame( # transform numpy array in Pandas datafraeme
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

X_train_processed_df
# ------

# TASK 8 — Verifica del preprocessing

print("=== PREPROCESSING - TASK 8: PROCESSED DATA CHECK ===")

import numpy as np

print("\n--- Shape ---")
print("X_train_processed:", X_train_processed.shape)
print("X_test_processed :", X_test_processed.shape)

print("\n--- Numero feature ---")
print(len(feature_names))

print("\n--- NaN nel training set ---")
print(np.isnan(X_train_processed).sum())

print("\n--- NaN nel test set ---")
print(np.isnan(X_test_processed).sum())

print("\n--- Valori unici nelle prime 10 feature ---")
for i, feature in enumerate(feature_names[:10]):
    print(feature, "→", np.unique(X_train_processed[:, i]))


# tutto come ci aspettavamo, nessune stranezze nelle trasformazioni e values delle colonne categoriche e numeriche

# TASK 9 — decidere se applicare lo scaling (probabilmente lo applicheremo)