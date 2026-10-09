# TREE_MODELS

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
    # drop="first" Non usiamo questa metrica perche andremo ad usare ensemble e non un modello lineare, e in questi e preferibile preservare tutte le features categoriali convertite in nr
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

# ----------

# TASK 10 — Random Forest

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    random_state=42 # Per rendere il risultato riproducibile
)

print(rf_model)
# non aggiungiamo class_weight, n_estimators, max_depth, vediamo a come si comporta la versione base del modello

rf_model.fit(
    X_train_processed,
    y_train
)

# modello allenato. Ora eseguiamo le predizioni sul test set

y_pred_rf = rf_model.predict(X_test_processed) 

print(y_pred_rf[:10])
print(y_pred_rf.shape) # -> 1499 osservazioni


# ---------
## Prima valutazione del Random Forest

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
) 

accuracy_rf = accuracy_score(y_test,y_pred_rf) # il test set del churn dello spit, paragonato alle predizioni del modello fatte su X_test
precision_rf = precision_score(y_test,y_pred_rf)
recall_rf = recall_score(y_test,y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("Accuracy :", accuracy_rf)
print("Precision:", precision_rf)
print("Recall   :", recall_rf)
print("F1-score :", f1_rf)

# ----------
# ROC-AUC e predict_proba()  

y_proba_rf = rf_model.predict_proba(X_test_processed)
print(y_proba_rf[:,1])

# -------

from sklearn.metrics import roc_auc_score # score nr

y_proba_rf = rf_model.predict_proba(X_test_processed)[:,1] ## prendiamo solo le % di decisione della classe 1 

roc_auc_rf = roc_auc_score(y_test,y_proba_rf) # verifica con i dati y_test e le prob di X_test

print("ROC-AUC:", roc_auc_rf) # 0.8226 -> il modello separa bene le classi

# -----------

# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_test, y_pred_rf)

print(cm_rf)


# Visualizzazione della Confusion Matrix

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["No Churn", "Churn"]
)

disp.plot()
plt.title("Random Forest - Confusion Matrix")
plt.show()

# -----------------

# TASK 11  --- XGBoost

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(
    X_train_processed,
    y_train
)

y_pred_xgb = xgb_model.predict(X_test_processed)
print(y_pred_xgb)
print(y_pred_xgb.shape)

print(y_pred_xgb)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

print("Accuracy :", accuracy_xgb)
print("Precision:", precision_xgb)
print("Recall   :", recall_xgb)
print("F1-score :", f1_xgb)

# ------
# Roc-auc score 

y_proba_xgb = xgb_model.predict_proba(X_test_processed)[:, 1]

roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)

print("ROC-AUC:", roc_auc_xgb) # 0.815

# ---------
# Confusion matrix

cm_xgb = confusion_matrix(y_test, y_pred_xgb) 
print(cm_xgb)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_xgb,
    display_labels=["No Churn", "Churn"]
)

disp.plot()
plt.title("XGBoost - Confusion Matrix")
plt.show()

# Recall legermente migliore del modello RadomForestClassifier

# ----------
# confronto delle metriche dei modelli ad alberi

baseline_results = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "Accuracy": [accuracy_rf, accuracy_xgb],
    "Precision": [precision_rf, precision_xgb],
    "Recall": [recall_rf, recall_xgb],
    "F1": [f1_rf, f1_xgb],
    "ROC-AUC": [roc_auc_rf, roc_auc_xgb]
})

print(baseline_results)

#  -------------

# Primo tuning per RandomForest: n_estimators 

rf_estimators_results = []

for n in [100,200,500]:

    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )

    model.fit(X_train_processed,y_train)

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:,1] # Solo probs affibiate alla colonna churn yes

    rf_estimators_results.append({
         "n_estimators": n,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }) 

rf_estimators_results = pd.DataFrame(rf_estimators_results)

print(rf_estimators_results)

# 200 alberi portano un risultato leggermente maggiore rispetto agli altri. 500 alberi ha portato il risultato peggiore su questo dataset, a prova che piu'
# alberi != a migliori performance. (Con xgboost, se aggiungiamo alberi senza controllo potremmo addirittura incappare nell'overfitting)

# -------

# Tuning di max_depth

rf_depth_results = []

for depth in [None, 5, 10, 15, 20]:

    model = RandomForestClassifier(
        n_estimators=200, # best tree nr so far
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    rf_depth_results.append({
        "max_depth": depth,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

rf_depth_results = pd.DataFrame(rf_depth_results)

# rf_depth_results
print(rf_depth_results)


# ----------

# min_samples_leaf

rf_leaf_results = []

for leaf in [1, 2, 5, 10]:

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=leaf,
        random_state=42
    )

    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    rf_leaf_results.append({
        "min_samples_leaf": leaf,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

rf_leaf_results = pd.DataFrame(rf_leaf_results)

print(rf_leaf_results)

# ------

# max_features

rf_features_results = []

for max_features in ["sqrt", "log2", None]:

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        max_features=max_features,
        random_state=42
    )

    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    rf_features_results.append({
        "max_features": max_features,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

rf_features_results = pd.DataFrame(rf_features_results)

print(rf_features_results)

# La scelta più equilibrata è sqrt.


# -----------

# Confrontiamo baseline vs Random Forest tuned

# baseline RF:
# F1 = 0.548673
# ROC-AUC = 0.822596

# Tuned RF:
# F1 = 0.586826
# ROC-AUC = 0.841421

# Creiamo una tabella ordinata:

rf_tuned_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)

rf_tuned_model.fit(X_train_processed, y_train)

y_pred_rf_tuned = rf_tuned_model.predict(X_test_processed)
y_proba_rf_tuned = rf_tuned_model.predict_proba(X_test_processed)[:, 1]

rf_tuned_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ],
    "Baseline": [
        accuracy_rf,
        precision_rf,
        recall_rf,
        f1_rf,
        roc_auc_rf
    ],
    "Tuned": [
        accuracy_score(y_test, y_pred_rf_tuned),
        precision_score(y_test, y_pred_rf_tuned),
        recall_score(y_test, y_pred_rf_tuned),
        f1_score(y_test, y_pred_rf_tuned),
        roc_auc_score(y_test, y_proba_rf_tuned)
    ]
})

print(rf_tuned_results)

# -------------

# Confusion Matrix del Random Forest tuned

cm_rf_tuned = confusion_matrix(
    y_test,
    y_pred_rf_tuned
)

print(cm_rf_tuned)

# il tuning ha migliorato entrambi i lati:
# TN +20
# FP -20
# FN -10
# TP +10

# ---------

# Visualizziamo la confusion matrix tuned

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf_tuned,
    display_labels=["No Churn", "Churn"]
)

disp.plot()
plt.title("Random Forest Tuned - Confusion Matrix")
plt.show()

# tuning random forest completato per ora
# ------------


# Inizio tuning XGBoost

# facciamo la stessa cosa fatta per Random Forest: prima analizziamo un parametro alla volta, mantenendo gli altri fissi.

# Partiamo da n_estimators, che controlla il numero di alberi del modello.

n_estimators = [100, 200, 500]

xgb_estimators_results = []

for n in [100, 200, 500]:

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n,
        random_state=42
    )

    model.fit(
        X_train_processed,
        y_train
    )

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    xgb_estimators_results.append({
        "n_estimators": n,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

xgb_estimators_results = pd.DataFrame(xgb_estimators_results)

xgb_estimators_results # 200 e' il nr di alberi favorito


# ----------

# XGBoost: max_depth


xgb_depth_results = []

for depth in [3, 5, 7, 10]:

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=depth,
        random_state=42
    )

    model.fit(
        X_train_processed,
        y_train
    )

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    xgb_depth_results.append({
        "max_depth": depth,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

xgb_depth_results = pd.DataFrame(xgb_depth_results)

xgb_depth_results # max_depth = 3 e' la piu' idonea 


# ------------

# learning_rate

xgb_lr_results = []

for lr in [0.01, 0.05, 0.1, 0.2]:

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=3,
        learning_rate=lr,
        random_state=42
    )

    model.fit(
        X_train_processed,
        y_train
    )

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    xgb_lr_results.append({
        "learning_rate": lr,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

xgb_lr_results = pd.DataFrame(xgb_lr_results)

xgb_lr_results # learning_rate = 0.05


# ---------

# min_child_weight

xgb_mcw_results = []

for mcw in [1, 3, 5, 10]:

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=mcw,
        random_state=42
    )

    model.fit(
        X_train_processed,
        y_train
    )

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    xgb_mcw_results.append({
        "min_child_weight": mcw,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

xgb_mcw_results = pd.DataFrame(xgb_mcw_results)

xgb_mcw_results # min_child_weight = 10


# ----------
# subsample

xgb_subsample_results = []

for subsample in [0.6, 0.8, 1.0]:

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=subsample,
        random_state=42
    )

    model.fit(
        X_train_processed,
        y_train
    )

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    xgb_subsample_results.append({
        "subsample": subsample,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

xgb_subsample_results = pd.DataFrame(xgb_subsample_results)

xgb_subsample_results # subsample=0.8


# ---------
# colsample_bytree

xgb_colsample_results = []

for colsample in [0.6, 0.8, 1.0]:

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=colsample,
        random_state=42
    )

    model.fit(
        X_train_processed,
        y_train
    )

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    xgb_colsample_results.append({
        "colsample_bytree": colsample,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

xgb_colsample_results = pd.DataFrame(xgb_colsample_results)

xgb_colsample_results # Scelta: colsample_bytree=1.0


# --------

# Tuning XGBoost finito per ora.

# Valutare XGBoost tuned

xgb_tuned_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42
)

xgb_tuned_model.fit(
    X_train_processed,
    y_train
)

y_pred_xgb_tuned = xgb_tuned_model.predict(X_test_processed)

y_proba_xgb_tuned = xgb_tuned_model.predict_proba(
    X_test_processed
)[:, 1]

xgb_tuned_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ],
    "Baseline": [
        accuracy_xgb,
        precision_xgb,
        recall_xgb,
        f1_xgb,
        roc_auc_xgb
    ],
    "Tuned": [
        accuracy_score(y_test, y_pred_xgb_tuned),
        precision_score(y_test, y_pred_xgb_tuned),
        recall_score(y_test, y_pred_xgb_tuned),
        f1_score(y_test, y_pred_xgb_tuned),
        roc_auc_score(y_test, y_proba_xgb_tuned)
    ]
})

# xgb_tuned_results
print(xgb_tuned_results)


# ---------
# Matrice di confusione di XGBoost tuned

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm_xgb_tuned = confusion_matrix(
    y_test,
    y_pred_xgb_tuned
)

print(cm_xgb_tuned)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_xgb_tuned,
    display_labels=["No Churn", "Churn"]
)

disp.plot()
plt.title("XGBoost Tuned - Confusion Matrix")
plt.show()


# ---

# Confronto diretto tra Random Forest tuned e XGBoost tuned


final_model_comparison = pd.DataFrame({
    "Model": [
        "Random Forest Tuned",
        "XGBoost Tuned"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred_rf_tuned),
        accuracy_score(y_test, y_pred_xgb_tuned)
    ],
    "Precision": [
        precision_score(y_test, y_pred_rf_tuned),
        precision_score(y_test, y_pred_xgb_tuned)
    ],
    "Recall": [
        recall_score(y_test, y_pred_rf_tuned),
        recall_score(y_test, y_pred_xgb_tuned)
    ],
    "F1": [
        f1_score(y_test, y_pred_rf_tuned),
        f1_score(y_test, y_pred_xgb_tuned)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, rf_tuned_model.predict_proba(X_test_processed)[:, 1]),
        roc_auc_score(y_test, y_proba_xgb_tuned)
    ]
})

final_model_comparison.sort_values(
    by="F1",
    ascending=False
)


# XGBoost tuned è il candidato migliore.

#                  Model  Accuracy  Precision    Recall        F1   ROC-AUC
# 1        XGBoost Tuned  0.805536   0.665563  0.537433  0.594675  0.848157
# 0  Random Forest Tuned  0.804116   0.666667  0.524064  0.586826  0.841421

# Non è ancora una decisione definitiva: i risultati provengono dal test set usato anche durante il tuning.


# -------

# Verificare la strategia di validazione

# Usiamo la cross-validation sul training set per selezionare i parametri, lasciando il test set per la valutazione finale.

from sklearn.pipeline import Pipeline