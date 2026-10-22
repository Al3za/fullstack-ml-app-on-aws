# TREE_MODELS

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


# TASK 2 — Pulizia e tipizzazione delle feature
# Partiamo da X, non da df_analysis, perché abbiamo già separato target e identificatore.

print("\n--- Data types prima della conversione ---")
print(X.dtypes)

print("\n--- Missing values prima della conversione ---")
print(X.isna().sum().sort_values(ascending=False))


# ---------
# TASK 3 — Separazione delle feature numeriche e categoriche
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

# TASK 5 — Encoding delle feature categoriche

from sklearn.preprocessing import OneHotEncoder

categorical_encoder = make_categorical_encoder() # preprocessing.py import

print(categorical_encoder)


# --------------

# TASK 6 — Costruire il ColumnTransformer. Fondamentale per la Pipeline che andremo a creare

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
# TASK 7 — Train/Test Split e Fit e trasformazione delle feature

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

# ------------
# Inizio tuning XGBoost

# facciamo la stessa cosa fatta per Random Forest: prima analizziamo un parametro alla volta, mantenendo gli altri fissi.

# Partiamo da n_estimators, che controlla il numero di alberi del modello:

#  n_estimators


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

# early_stopping_rounds


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

# XGBoost: max_depth

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


# ------------

# learning_rate

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
# subsample

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
# colsample_bytree

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

# Valutare le metriche del modello finale di XGBoost dopo il "tuning" manuale

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

# xgb_tuned_results
print(xgb_tuned_results)


# ---------
# Matrice di confusione di XGBoost finale dopo "tuned manuale"

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

# Confronto diretto tra i modelli "finali" Random Forest e XGBoost (entrambi tuned manualmente )


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


# XGBoost tuned è il modello con migliori prestazioni per il nostro caso.

#     Manual tuned Model  Accuracy   Precision  Recall   F1        ROC-AUC
# 1        XGBoost Tuned  0.805536   0.665563  0.537433  0.594675  0.848157
# 0  Random Forest Tuned  0.804116   0.666667  0.524064  0.586826  0.841421

# Non è ancora una decisione definitiva: i risultati provengono dal test set usato anche durante il tuning, il che rendono i risultati piu' "ottimistici"


# -------
# -------

# Verificare la strategia di validazione
# Dato che i risultati provengono dal test set usato anche durante il tuning, che rendono ¨
# i risultati delle prestazioni del modello piu' ottimistici, usiamo la cross-validation sul training set per selezionare i parametri, lasciando il test set per la valutazione finale.

# Nota : 
# Il test set e stato usato spesso durante il tuning dei modelli, risultando in una stima finale
# meno indipendente (ovvero piu' ottimistica). Il workflow corretto sarebbe suddividere i dati in
# train, test e validation, ma per questo progetto da portfolio non adoperiamo questo split.

# -------

# Crea la Pipeline per RandomForestClassifier

# Creiamo la pipeline dove inseriamo il ColumnTransformer creato in precedenza, in modo da no avere
# Data leakage durante la cross-validation k-fold

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

rf_pipeline = Pipeline([
    ("preprocessor",preprocessor),
    ("classifier",RandomForestClassifier(random_state=42)),
])

print(rf_pipeline)


# -------
# Preparare la ricerca degli iperparametri per Random Forest

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint # permette di definire intervalli di valori interi da esplorare,
# anziché elencare ogni valore possibile.

# -----
# Preparare la cross-validation

from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
# StratifiedKFold mantiene approssimativamente la stessa proporzione di clienti churn e non-churn in ogni fold.(73% / 26%) Si usa sempre con un dataset "sbilanciato"

# -------
# 4. Definire lo spazio di ricerca

rf_param_distributions = {
    "classifier__n_estimators": randint(100, 501),
    "classifier__max_depth": [None, 5, 10, 15, 20, 30],
    "classifier__min_samples_split": randint(2, 21),
    "classifier__min_samples_leaf": randint(1, 11),
    "classifier__max_features": ["sqrt", "log2", None]
}

# classifier__.. = il name del passagio definito nella pipeline. 
# il doppio underscore indica a scikit-learn che il parametro appartiene al classificatore interno alla pipeline.

# --- 
# Configurare RandomizedSearchCV

rf_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_param_distributions,
    n_iter=30,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    refit=True,
    verbose=1,
    return_train_score=False
)

print(rf_search)


# ------
# Eseguire il tuning della Random Forest


rf_search.fit(X_train, y_train)

# -------
# Mostra iperparametri, F1 medio per fold e pipeline

print("Migliori iperparametri:")
print(rf_search.best_params_)

print(f"\nMiglior F1 medio in CV: {rf_search.best_score_:.4f}")
# Miglior F1 medio in CV: 0.5820


best_rf_pipeline = rf_search.best_estimator_

print("\nPipeline selezionata:")
print(best_rf_pipeline)

# --------

# Misurare la stabilità dei risultati


cv_results_rf = pd.DataFrame(rf_search.cv_results_)

best_idx = rf_search.best_index_

print("F1 medio:", round(cv_results_rf.loc[best_idx, "mean_test_score"], 4))
print("Deviazione standard F1:", round(cv_results_rf.loc[best_idx, "std_test_score"], 4))

print("\nF1 sui singoli fold:")
for i in range(5):
    score = cv_results_rf.loc[best_idx, f"split{i}_test_score"]
    print(f"Fold {i + 1}: {score:.4f}")


# --------

# Ulteriore tuning manuale su Random forest di alcuni parametri sulla baseline dei parametri ottenuti con RandomizedSearchCV:

# max_features: 

from sklearn.base import clone # non modifichiamo ripetutamente la pipeline, creiamo invece una nuova pipeline per ogni configurazione, riutilizzando il preprocessor 

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

configs = {
    "None": None,
    "sqrt": "sqrt",
    "log2": "log2"
}

results = []

for name, max_features in configs.items():
    model = RandomForestClassifier(
        n_estimators=269,
        max_depth=10,
        min_samples_split=3,
        min_samples_leaf=6,
        max_features=max_features,
        random_state=42,
        n_jobs=1
    )

    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", model)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "f1": "f1",
            "precision": "precision",
            "recall": "recall"
        },
        n_jobs=-1
    )

    results.append({
        "configurazione": name,
        "F1 medio": scores["test_f1"].mean(),
        "F1 std": scores["test_f1"].std(),
        "Precision media": scores["test_precision"].mean(),
        "Recall medio": scores["test_recall"].mean()
    })

results_df = pd.DataFrame(results)
print(results_df.round(4).to_string(index=False))

# ------

# min_samples_split e min_samples_leaf: 


# Stessa cross-validation utilizzata nell'esperimento precedente
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Valori da confrontare
split_values = [3, 5, 10, 15]
leaf_values = [2, 4, 6, 10]

results = []

for min_split in split_values:
    for min_leaf in leaf_values:

        model = RandomForestClassifier(
            n_estimators=269,
            max_depth=10,
            min_samples_split=min_split, # 6 queste combinazioni portano a miglior recal, f1 e precision
            min_samples_leaf=min_leaf, # 15
            max_features=None,
            random_state=42,
            n_jobs=1
        )

        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", model)
        ])

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "f1": "f1",
                "precision": "precision",
                "recall": "recall"
            },
            n_jobs=-1
        )

        results.append({
            "min_samples_split": min_split,
            "min_samples_leaf": min_leaf,
            "F1 medio": scores["test_f1"].mean(),
            "F1 std": scores["test_f1"].std(),
            "Precision media": scores["test_precision"].mean(),
            "Recall medio": scores["test_recall"].mean()
        })

# Ordiniamo le configurazioni per F1 medio decrescente
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 medio",
    ascending=False
).reset_index(drop=True)

print(results_df.round(4).to_string(index=False))



# -------
# -------
# Consolizadione e Valutazione finale della Random Forest


rf_final = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", RandomForestClassifier(
        n_estimators=269,
        max_depth=10,
        min_samples_split=15,
        min_samples_leaf=6,
        max_features=None,
        random_state=42,
        n_jobs=-1
    ))
])

# Addestramento esclusivamente sul training set
rf_final.fit(X_train, y_train)

print("Random Forest addestrata correttamente.")

# ------
from sklearn.metrics import (
    classification_report
)


# Predizioni sul test set
y_pred = rf_final.predict(X_test)

# Probabilità di abbandono, utili per ROC-AUC
y_proba = rf_final.predict_proba(X_test)[:, 1]

# Metriche principali
metrics_rf = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-score": f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_proba)
}

metrics_rf_df = pd.DataFrame(
    metrics_rf.items(),
    columns=["Metrica", "Valore"]
)

print(metrics_rf_df.round(4).to_string(index=False))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No churn", "Churn"],
    zero_division=0
))

# Random Forest Conclusa.

# Importante_:
# Ricordiamo che il test set era già stato consultato durante il tuning manuale precedente:
# perciò questo confronto è utile per comprendere il comportamento dei modelli, ma non è una prova 
# indipendente della loro superiorità.

# -----------
# -----------
# -----------
# -----------
# -----------

# Preparare la pipeline XGBoost

# Ricordiamo che La pipeline applica il preprocessing e poi addestra XGBoost

xgb_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    ))
])

print("Pipeline XGBoost creata correttamente.")
print(xgb_pipeline)


# -----------
# Definiamo la ricerca degli iperparametri con il metodo cross-validation(oof) e RandomizedSearchCV.
# In questo modo si valutano le metriche direttamente sul train set, lasciando il test set "indipendente" per il test finale.
# Cosi' non usiamo il test set per analizare il miglioramento delle metriche del modello ad ogni iperparametro modificato,
# perche questo rende il test set "meno indipendente", portando le predizioni a risultare piu' ottimistiche  

from scipy.stats import randint, uniform, loguniform

xgb_param_distributions = {
    "classifier__n_estimators": randint(100, 501),
    "classifier__max_depth": randint(2, 8),
    "classifier__learning_rate": loguniform(0.01, 0.2),
    "classifier__min_child_weight": randint(1, 11),
    "classifier__subsample": uniform(0.6, 0.4),
    "classifier__colsample_bytree": uniform(0.6, 0.4),
    "classifier__gamma": uniform(0, 2)
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

xgb_search = RandomizedSearchCV(
    estimator=xgb_pipeline, # trasforma le colonne categoriche con OneHotEncoder
    param_distributions=xgb_param_distributions, # gli intervalli degli iperparametri che vogliamo analizzare per ottenere il miglior f1 score, come stabilito in basso
    n_iter=30,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    refit=True,
    verbose=1,
    return_train_score=False
)

xgb_search.fit(X_train,y_train)


print("Migliori parametri:")
print(xgb_search.best_params_)

print("\nMiglior F1 medio in cross-validation:")
print(round(xgb_search.best_score_, 4))

# ---------
# Ulteriore tuning "manuale" su XGBoost di alcuni parametri sulla baseline dei parametri ottenuti con RandomizedSearchCV:

# min_child_weight: 

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Parametri migliori trovati da RandomizedSearchCV
best_params = xgb_search.best_params_.copy()

# Rimuoviamo il prefisso della Pipeline per costruire XGBClassifier
xgb_params = {
    key.replace("classifier__", ""): value
    for key, value in best_params.items()
}

# controlla se i parametri sono corretti
print(f"check params: {xgb_params} ")

child_weight_values = [1, 3, 5, 7, 10, 12]
results_xgb_manual = []

for child_weight in child_weight_values:

    params = xgb_params.copy()
    params["min_child_weight"] = child_weight

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        **params # i params scelti tramite Random
    )

    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", model)
    ])

# Eseguiamo validazione oof con gli iperparametri trovati con RandomizedSearchCv, e facendo ulteriore tuning manuale su child_weight
    scores = cross_validate( 
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "f1": "f1",
            "precision": "precision",
            "recall": "recall"
        },
        n_jobs=-1
    )

    results_xgb_manual.append({
        "min_child_weight": child_weight,
        "F1 medio": scores["test_f1"].mean(),
        "F1 std": scores["test_f1"].std(),
        "Precision media": scores["test_precision"].mean(),
        "Recall medio": scores["test_recall"].mean()
    })

results_xgb_manual_df = pd.DataFrame(results_xgb_manual)

results_xgb_manual_df = results_xgb_manual_df.sort_values(
    by="F1 medio",
    ascending=False
).reset_index(drop=True)

print(results_xgb_manual_df.round(4).to_string(index=False))

# min_child_weight=1 rimane la configurazione migliore

# -------
# -------# 

# 1. Consolidiamo e addestriamo il modello finale

# Utilizziamo direttamente xgb*search.best_estimator*, che contiene già la pipeline con il preprocessing
# e i migliori parametri ed è già stato riaddestrato su tutto X_train grazie a refit=True.

# Modello XGBoost consolidato
xgb_final = xgb_search.best_estimator_

print("Parametri del modello finale:")
print(xgb_final.named_steps["classifier"].get_params())

print("\nModello XGBoost consolidato correttamente.")


# ------
# 2. Calcoliamo metriche e confusion matrix
y_pred_xgb = xgb_final.predict(X_test) # La pipeline si occuma delle trasformazioni delle colonne di X_test 

# Probabilità della classe positiva: Churn = 1
y_proba_xgb = xgb_final.predict_proba(X_test)[:,1] # possibile grazie alla metrica xgboost = objective="binary:logistic"

# Metriche
metrics_xgb = {
    "Accuracy": accuracy_score(y_test,y_pred_xgb),
    "Precision": precision_score(y_test, y_pred_xgb, zero_division=0),
    "Recall": recall_score(y_test, y_pred_xgb, zero_division=0),
    "F1-score": f1_score(y_test, y_pred_xgb, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test,y_proba_xgb)
}

metrics_xgb_df = pd.DataFrame(
    metrics_xgb.items(),
    columns=["Metrica", "Valore"]
)

print("METRICHE XGBOOST")
print(metrics_xgb_df.round(4).to_string(index=False))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred_xgb))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred_xgb,
    target_names=["No churn", "Churn"],
    zero_division=0
))


# XGBoost vince ancora e si candida come modello favorito sopra Random Forest

# -----

# Abbassiamo la soglia decisionale per aumentare i TP al minor costo di FP possibile.

# Scegliamo la soglia usando le predizioni out-of-fold

# Generiamo le probabilità out-of-fold


from sklearn.model_selection import cross_val_predict


# Ogni osservazione del training set viene predetta
# da un modello che non ha visto quell'osservazione
oof_proba = cross_val_predict(
    clone(xgb_search.best_estimator_), # clone() crea una copia non addestrata del modello, mantenendo gli iperparametri selezionati. # cross_val_predict avvia addestramento del modello vergine clonato sui fold previsti dalla cross-validation.
    X_train,
    y_train,
    cv=cv,
    method="predict_proba", # Valutazione con predizioni OOF
    n_jobs=-1
)[:, 1]

print("Predizioni out-of-fold generate:", len(oof_proba)) # lunghezza dei dati di taining

# ----
# confrontare le soglie


thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

threshold_results = []

for threshold in thresholds:
    y_pred_threshold = (oof_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_train,
        y_pred_threshold,
        labels=[0, 1]
    ).ravel()

    threshold_results.append({
        "Soglia": threshold,
        "Recall": recall_score(y_train, y_pred_threshold),
        "Precision": precision_score(
            y_train, y_pred_threshold, zero_division=0
        ),
        "F1": f1_score(y_train, y_pred_threshold),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    })

threshold_df = pd.DataFrame(threshold_results)

print(threshold_df.round(4).to_string(index=False))

# La soglia 0.35 offre il miglior F1 tra quelle testate e recupera più clienti a rischio rispetto
# a 0.50, contenendo parte dei falsi positivi rispetto alle soglie più basse.

# ------

# Il prossimo passo e' applicare la soglia al test set:

# Recall almeno del 70%, cercando di contenere gli FP

threshold_candidates = threshold_df[
    threshold_df["Recall"] >= 0.70
].sort_values(
    by=["FP", "Soglia"],
    ascending=[True, False]
)

print(threshold_candidates.round(4).to_string(index=False))

# ancora La soglia 0.35 offre miglior trade-off 

# -------
# valutare la soglia selezionata su test set

# Probabilità di churn sul test set
y_test_proba = xgb_final.predict_proba(X_test)[:, 1]

# Soglia selezionata tramite OOF
threshold = 0.35
y_test_pred_035 = (y_test_proba >= threshold).astype(int)

# Metriche
print("Soglia:", threshold)
print("Accuracy:", round(accuracy_score(y_test, y_test_pred_035), 4))
print("Precision:", round(precision_score(y_test, y_test_pred_035), 4))
print("Recall:", round(recall_score(y_test, y_test_pred_035), 4))
print("F1-score:", round(f1_score(y_test, y_test_pred_035), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_test_proba), 4))

print("\nMatrice di confusione:")
print(confusion_matrix(y_test, y_test_pred_035))

print("\nClassification report:")
print(classification_report(y_test, y_test_pred_035))


# --------
# --------
# --------

# ora proviamo ad aggiungere il parametro scale_pos_weight


import pandas as pd

scale_pos_weights = [1.0, 1.5, 2.0, 2.8]

weight_results = []

for weight in scale_pos_weights:
    model = clone(xgb_search.best_estimator_)

    model.set_params(
        classifier__scale_pos_weight=weight
    )

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        },
        n_jobs=-1
    )

    weight_results.append({
        "scale_pos_weight": weight,
        "Precision_mean": scores["test_precision"].mean(),
        "Recall_mean": scores["test_recall"].mean(),
        "F1_mean": scores["test_f1"].mean(),
        "ROC_AUC_mean": scores["test_roc_auc"].mean(),
        "F1_std": scores["test_f1"].std()
    })

weight_results_df = pd.DataFrame(weight_results)

print( 
    weight_results_df
    .sort_values("F1_mean", ascending=False)
    .round(4)
    .to_string(index=False)
)


print('check sopra')
# Confronto dei modelli

model_comparison = pd.DataFrame([
    {
        "Modello": "Random Forest",
        "Soglia": 0.50,
        "Accuracy": 0.7956,
        "Precision": 0.6378,
        "Recall": 0.5321,
        "F1-score": 0.5802,
        "ROC-AUC": 0.8392
    },
    {
        "Modello": "XGBoost",
        "Soglia": 0.50,
        "Accuracy": 0.8048,
        "Precision": 0.6667,
        "Recall": 0.5294,
        "F1-score": 0.5902,
        "ROC-AUC": 0.8472
    },
    {
        "Modello": "XGBoost",
        "Soglia": 0.35,
        "Accuracy": 0.7800,
        "Precision": 0.5684,
        "Recall": 0.7112,
        "F1-score": 0.6318,
        "ROC-AUC": 0.8472
    }
])

print(model_comparison.round(4).to_string(index=False))

# XGBoost vinvce su Random Forest al momento.
# Da mensionare che i parametri class_weight="balanced"(RF) e scale_pos_weight(XGB) non sono stati testati, da testare in futuro
