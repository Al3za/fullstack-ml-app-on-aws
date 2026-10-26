"""
Exploratory manual hyperparameter tuning — Random Forest.

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


# Pulizia e tipizzazione delle feature
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

# Check Encoding del target

print("=== CHECK IMPORTED TARGET DATA ===")

print("\n--- Target dopo la conversione ---")
print(y.value_counts())

print("\n--- Target dtype ---")
print(y.dtype)

print("\n--- Valori mancanti ---")
print(y.isna().sum())

# ---------

# Encoding delle feature categoriche

from sklearn.preprocessing import OneHotEncoder # funzione OneHotEncoder importata

categorical_encoder = make_categorical_encoder() # preprocessing.py import

print(categorical_encoder)

# --------------

# Costruire il ColumnTransformer. Fondamentale per la Pipeline che andremo a creare

from sklearn.compose import ColumnTransformer # funzione ColumnTransformer importata

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
# ----------

# Inizio creazione del modello Random Forest Base

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

# modello allenato. Procediamo con le predizioni sul test set

y_pred_rf = rf_model.predict(X_test_processed) 

print(y_pred_rf[:10])
print(y_pred_rf.shape) # -> 1499 osservazioni


# ---------
## Prima valutazione di Random Forest base

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
) 
import matplotlib.pyplot as plt

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



cm_rf = confusion_matrix(y_test, y_pred_rf)

print(cm_rf)


# Visualizzazione della Confusion Matrix

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["No Churn", "Churn"]
)

disp.plot()
plt.title("Random Forest - Confusion Matrix")
plt.show()

# Blocco 1

# Confronto delle metriche di RF e XGBoost baseline e senza tuning sugli iperparametri

# accuracy_xgb = "",
precision_xgb = ""
recall_xgb = ""
f1_xgb = ""
roc_auc_xgb = ""

baseline_results = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "Accuracy": [accuracy_rf, accuracy_xgb], # trova il modo di inserire i dati di XGb 
    "Precision": [precision_rf, precision_xgb],
    "Recall": [recall_rf, recall_xgb],
    "F1": [f1_rf, f1_xgb],
    "ROC-AUC": [roc_auc_rf, roc_auc_xgb]
})

print(baseline_results)


# ---

# TUNING MANUALE 
# Cerchiamo gli Iperparametri che offrono il miglior trade-off tra precision e recall 

# Tuning n_estimator:
# n_estimators indica quanti alberi compongono la Random Forest. Più alberi generalmente rendono la
# stima più stabile, ma aumentano il costo computazionale, e molti alberi potrebbero portare ad overfitting

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

# 200 alberi portano un risultato leggermente maggiore rispetto alle altre opzioni. 500 alberi ha portato il risultato peggiore su questo dataset, a prova che piu'
# alberi != a migliori performance. ( Una crescita incontrollata di alberi potrebbe portare all'overfitting)

# -------

# Tuning di max_dept

# max_dept definisce quanti livelli di decisioni successive può fare un albero. 
# Troppo basso = modello semplice. Troppo alto = Rischio overfitting

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
# min_samples_leaf indica il numero minimo di campioni che devono rimanere in una foglia (leaf) dell'albero.
# Una foglia è un nodo finale dell'albero, dove l'albero smette di fare ulteriori divisioni e produce una previsione.

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
# controlla quante feature vengono considerate come candidate quando un albero deve effettuare uno split

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

# La scelta più equilibrata in questo caso risulta sqrt.

# -----------
# -----------

# Random Forest baseline vs Random Forest tuned

# Definiamo il modello finale di RF tuned manualmente e compariamolo al modello base:

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

# Creiamo una tabella ordinata
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

# Visualiziamo la confusion_matrix:
 
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf_tuned,
    display_labels=["No Churn", "Churn"]
)

disp.plot()
plt.title("Random Forest Tuned - Confusion Matrix")
plt.show()

# tuning random forest completato per ora

# Verdetto: 

# Nel modello tuned vediamo un netto miglioramento delle metriche F1 e ROC-AUC 

# baseline RF:
# F1 = 0.548673
# ROC-AUC = 0.822596

# Tuned RF:
# F1 = 0.586826
# ROC-AUC = 0.841421
 
# Il tuning ha migliorato entrambi i lati della confusion matrix:
# TN +20
# FP -20
# FN -10
# TP +10


# Confronto diretto tra i modelli "finali" Random Forest e XGBoost (entrambi tuned manualmente )


# XGBoost tuned è il modello con migliori prestazioni per il nostro caso.

#     Manual tuned Model  Accuracy   Precision  Recall   F1        ROC-AUC
# 1        XGBoost Tuned  0.805536   0.665563  0.537433  0.594675  0.848157
# 0  Random Forest Tuned  0.804116   0.666667  0.524064  0.586826  0.841421

# Non è ancora una decisione definitiva: i risultati provengono dal test set usato anche durante il tuning, il che rendono i risultati piu' "ottimistici"
