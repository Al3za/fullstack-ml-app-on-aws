"""
Exploratory manual hyperparameter tuning — XGBoost.

These initial experiments used the test set both for early stopping
and for comparing hyperparameter configurations.

Using the test set in this way introduces information leakage into
model selection: the stopping iteration and the selected hyperparameters
are influenced by test-set performance.

The results are retained for educational and exploratory purposes only.
They are not used as the final estimate of model generalization.

The final model selection is performed using cross-validation on the
training set, with preprocessing fitted separately within each fold.
The test set is reserved for the final evaluation.
"""

import pandas as pd
from dataset import df_analysis, X, y, numeric_features, binary_features, categorical_features
from preprocessing import make_categorical_encoder, make_preprocessor
from data_split import X_train, X_test, y_train, y_test

print("=== DUBBLE CHECK IMPORTED X,y DATA ===")

print("\n--- X ---")
print(X.shape)

print("\n--- y ---")
print(y.shape)

print("\n--- Feature utilizzate ---")
print(X.columns.tolist()) # crea un list/array delle colonne di X. Id e' stato tolto correttamente

print("\n--- Target ---")
print(y.name)


#  Pulizia e tipizzazione delle feature
# Partiamo da X, non da df_analysis, perché abbiamo già separato target e identificatore.

print("\n--- Data types prima della conversione ---")
print(X.dtypes)

print("\n--- Missing values prima della conversione ---")
print(X.isna().sum().sort_values(ascending=False))


# ---------
# Separazione delle feature numeriche e categoriche
print("=== DUBBLE CHECK IMPORTED FEATURE GROUPS ===")

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

print("=== CHECK IMPORTED TARGET DATA ===")

print("\n--- Target dopo la conversione ---")
print(y.value_counts())

print("\n--- Target dtype ---")
print(y.dtype)

print("\n--- Valori mancanti ---")
print(y.isna().sum())

# ---------

# Encoding delle feature categoriche

from sklearn.preprocessing import OneHotEncoder

categorical_encoder = make_categorical_encoder() # preprocessing.py import

print(categorical_encoder)

# --------------

# Costruire il ColumnTransformer. Fondamentale per la Pipeline che andremo a creare

from sklearn.compose import ColumnTransformer

preprocessor = make_preprocessor(numeric_features, binary_features, categorical_features)

print(preprocessor)

print("=== DUBBLE CHECK PREPROCESSOR  ===")
print("\n--- Feature totali ---")
print(len(numeric_features) + len(binary_features) + len(categorical_features)) # 19

#

print("\n--- Preprocessor configurato ---")
print(preprocessor.transformers) ## "mostra la configurazione dei trasformatori che ho definito."


# --------
# Controlla Train/Test Split e Fit e trasformazione delle feature

from sklearn.model_selection import train_test_split

print("DUBBLE CHECK IMPORTED TRAIN / TEST SPLIT ===")

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

# Verifica del preprocessing

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


# Trasformazioni delle colonne categoriche e numeriche avvenute correttamente

# ----------

# Inizio creazione del modello XGBoost Base

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix, 
    ConfusionMatrixDisplay
) 

import matplotlib.pyplot as plt

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(
    X_train_processed,
    y_train
)

# modello allenato. Procediamo con le predizioni sul test set

y_pred_xgb = xgb_model.predict(X_test_processed)
print(y_pred_xgb)
print(y_pred_xgb.shape)

print(y_pred_xgb)

## Prima valutazione di XGBoost base

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

# ---
# Confronto delle metriche di RF e XGBoost baseline e senza tuning sugli iperparametri


# accuracy_xgb = "",
precision_xgb = ""
recall_xgb = ""
f1_xgb = ""
roc_auc_xgb = ""

baseline_results = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "Accuracy": [accuracy_rf, accuracy_xgb], # trova il modo di inserire i dati di RF baseline 
    "Precision": [precision_rf, precision_xgb],
    "Recall": [recall_rf, recall_xgb],
    "F1": [f1_rf, f1_xgb],
    "ROC-AUC": [roc_auc_rf, roc_auc_xgb]
})

print(baseline_results)

# ---

# TUNING MANUALE 
# Cerchiamo gli Iperparametri che offrono il miglior trade-off tra precision e recall 

# Tuning n_estimators:
# n_estimators indica quanti alberi compongono la Random Forest. Più alberi generalmente rendono la
# stima più stabile, ma aumentano il costo computazionale, e molti alberi potrebbero portare ad overfitting

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

# early_stopping_rounds. 
# early_stopping_rounds interrompe l'addestramento se la metrica monitorata non migliora per 30 round consecutivi.
# Usa il test set per monitorare se logloss non migliora l'addestramento. Questo e' un importante data lekage sul test set
# per questo usiamo questa configurazione per monitorare la prestazione del modello solo come esperimento.

xgb_early_stopping_rounds_results = []

for rounds in [10, 20, 30]:

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        early_stopping_rounds= rounds,
        random_state=42
    )

    model.fit(
        X_train_processed,
        y_train,
        
        
    )

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    xgb_early_stopping_rounds_results.append({
        "early_stopping_rounds": rounds,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

xgb_early_stopping_rounds_results = pd.DataFrame(xgb_early_stopping_rounds_results)

xgb_early_stopping_rounds_results # 30 rounds e' il nr candidato


# ----------

# max_depth:
# max_dept come in RF, definisce quanti livelli di decisioni successive può fare un albero. 
# Troppo basso = modello semplice. Troppo alto = Rischio overfitting

xgb_depth_results = []

for depth in [3, 5, 7, 10]:

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        early_stopping_rounds=30,
        max_depth=depth,
        random_state=42
    )

    model.fit(
        X_train_processed,
        y_train,
        eval_set=[(X_test_processed, y_test)],
        verbose=False
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


# ----------

# learning_rate:

# Controlla quanto ogni nuovo albero contribuisce alla correzione del modello precedente.
# learning rate basso → apprendimento più graduale -> il training puo' essere piu' lento ma migliore per la generalizzazione
# learning rate alto → apprendimento più rapido, ma potenzialmente meno stabile, puo' adattarsi troppo hai dati di traning (overfitting)

xgb_lr_results = []

for lr in [0.01, 0.05, 0.1, 0.2]:

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        early_stopping_rounds=30,
        max_depth=3,
        learning_rate=lr,
        random_state=42
    )

    model.fit(
        X_train_processed,
        y_train,
        eval_set=[(X_test_processed, y_test)],
        verbose=False
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
# Stabilisce quanto “peso” deve avere un nodo figlio perché XGBoost possa creare uno split.
# “Non creare un nuovo split se il nodo figlio risultante è troppo poco significativo. (Cioe' se lo split e' troppo specifico e riguarda solo pochi osservazioni(clienti))

xgb_mcw_results = []

for mcw in [1, 3, 5, 10]:

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        early_stopping_rounds=30,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=mcw,
        random_state=42
    )

    model.fit(
        X_train_processed,
        y_train,
        eval_set=[(X_test_processed, y_test)],
        verbose=False
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
# ----------

# subsample:
# Questo parametro stabilisce quale frazione delle righe del training set viene utilizzata per costruire ogni albero.
# valore = 1.0 → usa tutte le righe
# valori inferiori → introducono maggiore casualità

xgb_subsample_results = []

for subsample in [0.6, 0.8, 1.0]:

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        early_stopping_rounds=30,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=subsample,
        random_state=42
    )

    model.fit(
        X_train_processed,
        y_train,
        eval_set=[(X_test_processed, y_test)],
        verbose=False
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
# colsample_bytree:

# concettualmente simile a subsample, ma la differenza e' che colsample_bytree → campiona feature

xgb_colsample_results = []

for colsample in [0.6, 0.8, 1.0]:

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,      
        early_stopping_rounds=30,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=colsample,
        random_state=42
    )

    model.fit(
        X_train_processed,
        y_train,
        eval_set=[(X_test_processed, y_test)],
        verbose=False
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

# Tuning "manuale" su XGBoost  finito per ora.

# Valutare le metriche del modello base finale di XGBoost dopo il "tuning" manuale, e confrontiamole con il modello base

xgb_tuned_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=200,
    early_stopping_rounds=30,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42
)

xgb_tuned_model.fit(
    X_train_processed,
    y_train,
    eval_set=[(X_test_processed, y_test)],
    verbose=False   
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

print(xgb_tuned_results)

# Visualiziamo la confusion_matrix:
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

# Verdetto: 

# Nel modello tuned vediamo un netto miglioramento delle metriche F1 e ROC-AUC 

# baseline XGBoost:
# F1 = 0.5442..
# ROC-AUC = 0.8152..

# Tuned XGBoost:
# F1 = 0.594675
# ROC-AUC = 0.848157

# ---

# Confronto diretto tra i modelli "finali" Random Forest e XGBoost (entrambi tuned manualmente )


# XGBoost tuned è il modello con migliori prestazioni per il nostro caso.

#     Manual tuned Model  Accuracy   Precision  Recall   F1        ROC-AUC
# 1        XGBoost Tuned  0.805536   0.665563  0.537433  0.594675  0.848157
# 0  Random Forest Tuned  0.804116   0.666667  0.524064  0.586826  0.841421

# Non è ancora una decisione definitiva: i risultati provengono dal test set usato anche durante il tuning, il che rendono i risultati piu' "ottimistici"
