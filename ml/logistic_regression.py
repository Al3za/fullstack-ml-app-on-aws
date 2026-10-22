# Logistic Regression

import pandas as pd

DATA_PATH = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH) # i dati raw non modificati

df_analysis = df.copy() # dati modificati

# Transform TotalCharges empty data "" to 0 to perform  on the column:

df_analysis["TotalCharges"] = pd.to_numeric( # Convert object type to a numeric type.
    df_analysis["TotalCharges"],
    errors="coerce" # sets the "" values to NAN
)

df_analysis['TotalCharges'] = df_analysis['TotalCharges'].fillna(0)

print((df_analysis['TotalCharges'] == 0).sum())

print("=== PREPROCESSING - TASK 1: FEATURE PREPARATION ===")

target = "Churn"
id_column = "customerID"

X = df_analysis.drop(columns=[target,id_column]) # id non ci serve per addestrare il modello, e targhet lo separiamo semplicemente dalle variabili indipendenti
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
    sparse_output=False, # sparse_output=True -> utile quando abbiamo molte categorie (in questo caso abbiamo al massimo 3)
    # drop="first" # elimina la prima colonna per evitare la collinearita' perfetta tra le variabili categoriche trasformate con OHE (dummy variable trap)
    # abilita drop="first" dopo aver testato il tuning sulla regolarizazione L1, L2
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

# TASK 9 — Creiamo il preprocessing con scaling

from sklearn.preprocessing import StandardScaler

# Nuovo preprocessing dedicato alla Logistic Regression
preprocessor_lr = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        ("binary", "passthrough", binary_features),
        ("categorical", categorical_encoder, categorical_features)
    ],
    remainder="drop",
    verbose_feature_names_out=True
)

print("Preprocessor per Logistic Regression creato.")


# ----
# Verificare il nuovo preprocessing

# Applichiamo il preprocessing al training e al test set
X_train_lr = preprocessor_lr.fit_transform(X_train)
X_test_lr = preprocessor_lr.transform(X_test)

# Recuperiamo i nomi delle feature
feature_names_lr = preprocessor_lr.get_feature_names_out()

print("=== TASK 9: CHECK SCALING ===")
print("\n--- Shape ---")
print("X_train_lr:", X_train_lr.shape)
print("X_test_lr :", X_test_lr.shape)

print("\n--- Numero feature ---")
print(len(feature_names_lr))

print("\n--- Valori NaN ---")
print("Training:", X_train_lr.isna().sum().sum()
      if hasattr(X_train_lr, "isna") else __import__("numpy").isnan(X_train_lr).sum())
print("Test:", X_test_lr.isna().sum().sum()
      if hasattr(X_test_lr, "isna") else __import__("numpy").isnan(X_test_lr).sum())

print("\n--- Statistiche feature numeriche scalate (training) ---")

for feature in ["numeric__tenure",
                "numeric__MonthlyCharges",
                "numeric__TotalCharges"]:
    idx = list(feature_names_lr).index(feature)
    print(
        f"{feature}: mean={np.mean(X_train_lr[:, idx]):.4f}, "
        f"std={np.std(X_train_lr[:, idx]):.4f}"
    )


# ------

# Pipeline e Addestramento della Logistic Regression
# Usiamo il preprocessing dentro la pipeline


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

lr_baseline = Pipeline([
    ("preprocessor", preprocessor_lr),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

lr_baseline.fit(X_train, y_train)

print("Logistic Regression addestrata correttamente.")
print(
    "Numero di iterazioni:",
    lr_baseline.named_steps["classifier"].n_iter_
)


# ------

# Prima valutazione del modello baseline sul test set

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Predizioni sul test set
y_pred_lr = lr_baseline.predict(X_test) 
y_proba_lr = lr_baseline.predict_proba(X_test)[:, 1]

# Metriche
print("--- METRICHE LOGISTIC REGRESSION ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba_lr):.4f}")

# Matrice di confusione
print("\n--- MATRICE DI CONFUSIONE ---")
print(confusion_matrix(y_test, y_pred_lr))

# Report per classe
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(
    y_test,
    y_pred_lr,
    target_names=["No Churn", "Churn"],
    digits=4
))
# baseline di logistic Regression risulta migliore delle baseline Random forest e XGBoost. Quest'ultima ha un ROC-AUC leggermente piu' alto


# ------

# Analisi delle curve ROC e Precision-Recall

# Curva ROC:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Calcolo dei punti della curva ROC
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba_lr)
roc_auc = roc_auc_score(y_test, y_proba_lr)

# Grafico
plt.figure(figsize=(8, 6))
plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.4f})"
)
plt.plot([0, 1], [0, 1], linestyle="--", label="Classificatore casuale")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Curva ROC — Logistic Regression")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#----------

# Calcolo e grafico Precision-Recall
# Cerchiamo di ricavare il miglior trade-off tra recalla e precision 

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)

# Calcolo della curva Precision-Recall
precision_lr, recall_lr, thresholds_pr = precision_recall_curve(
    y_test,
    y_proba_lr
)

# Average Precision
ap_lr = average_precision_score(y_test, y_proba_lr)

# Grafico
plt.figure(figsize=(8, 6))
plt.plot(
    recall_lr,
    precision_lr,
    label=f"Logistic Regression (AP = {ap_lr:.4f})"
)

# Baseline: proporzione di clienti Churn nel test set
baseline_churn = (y_test == 1).mean()
plt.axhline(
    y=baseline_churn,
    linestyle="--",
    label=f"Baseline Churn = {baseline_churn:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall — Logistic Regression")
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Average Precision: {ap_lr:.4f}")
print(f"Baseline Churn:    {baseline_churn:.4f}")


# ------

# Ottimizzazione della soglia decisionale con il metodo Cross-validation OOF. 

# Usiamo un cross-validation per ottenere le prob dei clienti che secondo il modello faranno churn.
# Usiamo la cross-validation per non dover effettuare questi controlli sui dati di test, che useremo
# soltanto per la valutazione finale del modello, e per mantenere questi dati indipendenti

# Queste Probs poi le useremo per valutare la migliore threshold che ci aiutera' a ottenere il miglior
# Trade-off tra recall e precision 

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Pipeline indipendente per la cross-validation. (esente dai dati del fit precedente)
lr_cv = clone(lr_baseline)

# Cross-validation stratificata a 5 fold
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Probabilità out-of-fold della classe positiva (Churn = 1)
y_proba_oof_lr = cross_val_predict( 
    lr_cv, # La pipeline applica le trasformazioni (scaler, OHE) e allena il modello. Fit_transform sui gruppi di train, solo transform sul validation grup (per ogni fold)
    X_train,
    y_train,
    cv=cv, # crea i folds con principio' StratifiedKFold 
    method="predict_proba", # raccoglie le probabilita' dei 5 fold che i modelli hanno ottenuto dal "gruppo validazione" di ogni fold
    n_jobs=-1 # Usa tutti i CPU per velocizzare il processo
)[:, 1] # solo i dati delle prob delle predizioni sul churn=1 
 
print("Probabilità OOF calcolate correttamente.")
print("Numero di osservazioni:", len(y_proba_oof_lr))
print("Valori minimi e massimi:",
      round(y_proba_oof_lr.min(), 4), # il range delle probs. Da..
      round(y_proba_oof_lr.max(), 4)) # fino...
print("Valori NaN:", np.isnan(y_proba_oof_lr).sum())



# ------
# Confronto tra soglie candidate

thresholds = np.arange(0.20, 0.61, 0.05) # (start, stop, step)
results_lr = []

for threshold in thresholds:
    y_pred_oof = (y_proba_oof_lr >= threshold).astype(int) # .astype(int) ritorna 0 e 1 in base se >= threshold e' True(1) o False(0)

    tn, fp, fn, tp = confusion_matrix(
        y_train, y_pred_oof, labels=[0, 1] # usiamo y_train perche sono i dati che dobbiamo comparare con y_pred_oof, dato che le probs sono state ottenute con i dati di X_train e y_train  Non e' data leakage perche li usiamo solo per misurare la confusion matrix
    ).ravel()

    results_lr.append({
        "Threshold": round(threshold, 2),
        "Accuracy": accuracy_score(y_train, y_pred_oof),
        "Precision": precision_score(
            y_train, y_pred_oof, zero_division=0
        ),
        "Recall": recall_score(
            y_train, y_pred_oof, zero_division=0
        ),
        "F1-score": f1_score(
            y_train, y_pred_oof, zero_division=0
        ),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

threshold_results_lr = pd.DataFrame(results_lr)

print(
    threshold_results_lr.round(4).to_string(index=False)
)

# La soglia 0.35 è quella che fa' piu' al caso nostro, con recall  > 71% e al costo di relativamente ridotti FP


# ---------
# Ricerca più fine della soglia decisionale
# Esamineremo le soglie da 0.30 a 0.40, con incrementi di 0.01, mantenendo il criterio stabilito: Recall almeno pari al 70% e, 
# tra le soglie ammissibili, il minor numero di falsi positivi


thresholds_fine = np.arange(0.30, 0.401, 0.01) # ricerca piu' fine intorno la soglia candidata
results_fine_lr = []

# Calcola Precision, Recall, F1-score e matrice di confusione per ogni soglia:
for threshold in thresholds_fine:
    y_pred_oof = (y_proba_oof_lr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_train, y_pred_oof, labels=[0, 1]
    ).ravel()

    results_fine_lr.append({
        "Threshold": round(threshold, 2),
        "Precision": precision_score(
            y_train, y_pred_oof, zero_division=0
        ),
        "Recall": recall_score(
            y_train, y_pred_oof, zero_division=0
        ),
        "F1-score": f1_score(
            y_train, y_pred_oof, zero_division=0
        ),
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "TN": tn
    })

threshold_fine_df = pd.DataFrame(results_fine_lr)

# Mostriamo tutte le soglie esaminate
print(threshold_fine_df.round(4).to_string(index=False))

# Selezione delle soglie che rispettano il vincolo di Recall (> 70%)
eligible = threshold_fine_df[
    threshold_fine_df["Recall"] >= 0.70
]

if not eligible.empty:
    best_threshold_row = eligible.sort_values(
        ["FP", "Threshold"],
        ascending=[True, False]
    ).iloc[0]

    print("\n--- SOGLIA SELEZIONATA ---")
    print(best_threshold_row.round(4).to_string())
else:
    print("\nNessuna soglia soddisfa il vincolo di Recall >= 0.70.")

# la soglia 0.36 e' la nuova candidata

# ----

# Verifica della calibrazione delle probabilita' del modello.
# calcola il Brier score delle probabilità OOF e disegna la curva di calibrazione


import matplotlib.pyplot as plt

from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

# Brier score delle probabilità out-of-fold
brier_lr = brier_score_loss(y_train, y_proba_oof_lr)

# Riferimento: probabilità costante uguale alla prevalenza
prevalence_train = np.mean(y_train)
proba_baseline = np.full(len(y_train), prevalence_train)

# brier score con probabilita' costante(indiscriminante, stessa probabilità per tutti)
brier_baseline = brier_score_loss(
    y_train,
    proba_baseline
)

# Curva di calibrazione
fraction_positive, mean_predicted = calibration_curve(
    y_train,
    y_proba_oof_lr,
    n_bins=10,
    strategy="quantile"
)

# Grafico
plt.figure(figsize=(8, 6))

plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    label="Logistic Regression"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Calibrazione perfetta"
)

plt.xlabel("Probabilità media predetta")
plt.ylabel("Frazione reale di Churn")
plt.title("Curva di calibrazione — Logistic Regression")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Misura quanto il Brier Score della Logistic Regression migliora rispetto a una baseline che assegna a ogni cliente la stessa probabilità di churn, pari alla prevalenza osservata nel training set(26%)
print(f"Prevalenza Churn nel training set: {prevalence_train:.4f}")
print(f"Brier score Logistic Regression:  {brier_lr:.4f}")
print(f"Brier score baseline:             {brier_baseline:.4f}")

# Il modello Logistic Regression presenta una buona calibrazione delle probabilità

# ---
# Verificare class_weight="balanced"

# Prima di valutare la Logistic Regression alla soglia 0.36 sul test set, assegnamo un peso maggiore alla classe Churn per vedere se il modello aumenta le prestazioni:

lr_balanced = Pipeline([
    ("preprocessor", preprocessor_lr), # "preprocessor" = il ColumnTransformer
    ("classifier", LogisticRegression( # "classifier" = il classificatore
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ))
])

lr_balanced.fit(X_train, y_train)

print("Modello Logistic Regression con class_weight='balanced' addestrato.")
print("Classi:", lr_balanced.named_steps["classifier"].classes_)
print("Iterazioni:", lr_balanced.named_steps["classifier"].n_iter_)


# -----
# Confronto OOF: baseline vs class_weight="balanced"

# confrontiamo le probabilità out-of-fold dei due modelli

# Stessi fold per entrambi i modelli
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Probabilità OOF: Logistic Regression baseline
# (dati salvati su y_proba_oof_lr)

# Probabilità OOF: Logistic Regression "balanced"
y_proba_oof_balanced = cross_val_predict(
    clone(lr_balanced),
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

# Confronto alle soglie di default (0.50)
def calcola_metriche_oof(y_true, y_proba, threshold=0.50):
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "Brier score": brier_score_loss(y_true, y_proba)
    }

risultati_oof = pd.DataFrame({
    "Baseline": calcola_metriche_oof(y_train, y_proba_oof_lr),
    "Balanced": calcola_metriche_oof(y_train, y_proba_oof_balanced)
}).T

print(risultati_oof.round(4))

# class_weight="balanced" introduce un grande aumento di recall e di F1_score, ma anche un tangibile abbassamento di precision,e un legero aumento di Brier score


# ---------
# ---------

# esplorare le soglie OOF del modello balanced

thresholds = np.arange(0.30, 0.71, 0.02)

rows = []

for threshold in thresholds:
    y_pred = (y_proba_oof_balanced >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_train,
        y_pred,
        labels=[0, 1]
    ).ravel()

    rows.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_train, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_train, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_train, y_pred, zero_division=0
        ),
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "TN": tn
    })

threshold_table_balanced = pd.DataFrame(rows)

print(
    threshold_table_balanced.round(4).to_string(index=False)
    )

# ------
# la soglia balanced da considerare è 0.60 perche' sta in linea con i criteri scelti per i trees models ovvero
# 1) Un Recall almeno pari a 0.70
# 2) Scegliere la soglia con il minor numero di falsi positivi

# Confronto diretto con la baseline con soglia 0.36:

# - Balanced identifica 23 clienti in più che abbandonano.
# - Riduce i falsi negativi da 443 a 420
# - Al prezzo di 25 falsi positivi in più
# - L'F1-score migliora leggermente, da 0.6324 a 0.6370


#-------
#-------

# Valutare sul "test set" i due candidati con le soglie selezionate


# Probabilità dei modelli sul test set
y_proba_test_baseline = lr_baseline.predict_proba(X_test)[:, 1]
y_proba_test_balanced = lr_balanced.predict_proba(X_test)[:, 1]

# Soglie selezionate tramite OOF sul training set
threshold_baseline = 0.36
threshold_balanced = 0.60

y_pred_test_baseline = (
    y_proba_test_baseline >= threshold_baseline
).astype(int)

y_pred_test_balanced = (
    y_proba_test_balanced >= threshold_balanced
).astype(int)

def metriche_test(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

confronto_test = pd.DataFrame({
    "Baseline (0.36)": metriche_test(
        y_test, y_pred_test_baseline, y_proba_test_baseline
    ),
    "Balanced (0.60)": metriche_test(
        y_test, y_pred_test_balanced, y_proba_test_balanced
    )
}).T

print(confronto_test.round(4))

#  la Logistic Regression baseline con soglia 0.36 e' la  miglio canidata al momento, perché offre un F1-score leggermente superiore e meno falsi positivi,
#  mantenendo un recall vicino al 70%.




# -----
# RandomOverSampler su soglia standard 0.5 e senza Class_weight="balanced"


# Vediamo se riusciamo a migliorare le metriche con un metodo di OverSampling.

# il resampling deve avvenire all'interno di ogni fold di training per evitare che copie di dati si trovano nella sia parte usata per addestrare sia in quella usata per validare

# p.s useremo RandomOverSampler da solo, senza class_weight="balanced", per isolare l'effetto del resampling.

import imblearn
from imblearn.over_sampling import RandomOverSampler

# Costruiamo la pipeline

from imblearn.pipeline import Pipeline as ImbPipeline  # alias
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression

lr_ros = ImbPipeline([ # ImbPipeline fa' in modo che l'oversampling viene applicato solamente sui dati di train entro ogni folder (Sul validation set non vengono creati dati sintetici, vengono usati solo per fare le pred)
    ("preprocessor", preprocessor_lr),
    ("sampler", RandomOverSampler(random_state=42)), # aumenta la rappresentazione delle features appartenenti alla classe Churn = 1 duplicando esempi del training fold
    ("classifier", LogisticRegression( # addestra la Logistic Regression sui dati riequilibrati
        max_iter=1000,
        random_state=42
    ))
])

lr_ros.fit(X_train, y_train)

print("Pipeline addestrata correttamente.")
print("Classi:", lr_ros.classes_)
print(lr_ros)


# ---------
# Calcoliamo le probabilità OOF su soglia standard 0.5

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

y_proba_oof_ros = cross_val_predict(
    clone(lr_ros), # Ricorda che clone evita potenziali data lekage dopo dati del preprocessor delle colonne, applica solo le trasformazioni scaling e OHE sulle colonne stabilite, senza fare nuovamente il fit
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

print("Numero di probabilità OOF:", len(y_proba_oof_ros))
print("Valori mancanti:", np.isnan(y_proba_oof_ros).sum())
print("Probabilità minima:", round(y_proba_oof_ros.min(), 4))
print("Probabilità massima:", round(y_proba_oof_ros.max(), 4))


# -----
# Confronto OOF tra i modelli alla soglia 0.50


def valuta_oof(y_true, y_proba, soglia=0.50):
    y_pred = (y_proba >= soglia).astype(int)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "Brier score": brier_score_loss(y_true, y_proba)
    }

confronto_oof = pd.DataFrame({
    "Baseline": valuta_oof(y_train, y_proba_oof_lr),
    "Balanced": valuta_oof(y_train, y_proba_oof_balanced),
    "RandomOverSampler": valuta_oof(y_train, y_proba_oof_ros)
}).T

print(confronto_oof.round(4))

# RandomOverSampler e class_weight="balanced" producono risultati quasi identici alla soglia di 0.50.
# rispetto alla baseline, aumenta il Recall e F1-score, ma precision scende considerevolmente, e scende lievemente anche Brier score  

# ---------

# Tuning della soglia su RandomOverSampler model:

# Applichiamo il criterio già stabilito: raggiungere un recall di almeno 0.70 e, tra le soglie ammissibili, scegliere quella con meno falsi positivi.

risultati_ros = []

for soglia in [round(x, 2) for x in __import__("numpy").arange(0.30, 0.701, 0.01)]:
    y_pred = (y_proba_oof_ros >= soglia).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_train, y_pred, labels=[0, 1]
    ).ravel()

    risultati_ros.append({
        "Threshold": soglia,
        "Precision": precision_score(y_train, y_pred, zero_division=0),
        "Recall": recall_score(y_train, y_pred, zero_division=0),
        "F1-score": f1_score(y_train, y_pred, zero_division=0),
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "TN": tn
    })

tabella_ros = pd.DataFrame(risultati_ros)

ammissibili_ros = tabella_ros[
    tabella_ros["Recall"] >= 0.70
]

scelta_ros = ammissibili_ros.sort_values(
    ["FP", "Threshold"],
    ascending=[True, False]
).iloc[0]

print("Soglia selezionata secondo il criterio stabilito:")
print(scelta_ros.to_frame("Valore").round(4))

# soglia scelta = 0.61
# Il modello soddisfa il vincolo di recall almeno pari al 70%. Inoltre, produce 778 falsi positivi,
#  leggermente meno dei 780 della baseline alla soglia 0.36 e degli 805 del modello Balanced alla soglia 0.60.

# -----

# Valutazione sul test set

# valutiamo RandomOverSampler sul test set alla soglia 0.61

# Probabilità sul test set
y_proba_test_ros = lr_ros.predict_proba(X_test)[:, 1]

# Soglia selezionata tramite OOF sul training set
threshold_ros = 0.61

y_pred_test_ros = (
    y_proba_test_ros >= threshold_ros
).astype(int) # 0/1

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred_test_ros,
    labels=[0, 1]
).ravel()

risultati_test_ros = {
    "Accuracy": accuracy_score(y_test, y_pred_test_ros),
    "Precision": precision_score(
        y_test, y_pred_test_ros, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred_test_ros, zero_division=0
    ),
    "F1-score": f1_score(
        y_test, y_pred_test_ros, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(y_test, y_proba_test_ros),
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp
}

print(pd.Series(risultati_test_ros).round(4))

# RandomOverSampler ottiene i valori migliori, seppur di poco, per accuracy, precision, 
# F1-score e ROC-AUC. Tuttavia, identifica 261 clienti che abbandonano, contro i 262 della baseline
# e i 264 del modello Balanced.


# ------

# OverSampling con SMOTENC - Generazione di campioni sintetici
# SMOTENC è probabilmente la scelta più appropriata perché combina variabili numeriche e categoriche.

print("Dimensioni X_train:", X_train.shape)
print("Colonne categoriche presenti:", set(categorical_features).issubset(X_train.columns))
print("Colonne numeriche presenti:", set(numeric_features).issubset(X_train.columns))
print("Colonna binaria presente:", set(binary_features).issubset(X_train.columns))
print("Valori mancanti in X_train:", X_train.isna().sum().sum())
print("Valori vuoti:", X_train.map(
    lambda x: isinstance(x,str) and x.strip() == ""
).sum().sum())

print("\nTipi delle variabili categoriche:")
print(X_train[categorical_features].dtypes)

# Identifichiamo le colonne categoriche per SMOTENC

categorical_indices= [X_train.columns.get_loc(col)
       for col in categorical_features
    ] 

print("Indici categorici per SMOTENC:", categorical_indices)


# Costruiamo la pipeline SMOTENC

from imblearn.over_sampling import SMOTENC
# from imblearn.pipeline import Pipeline as ImbPipeline
# from sklearn.linear_model import LogisticRegression

lr_smotenc = ImbPipeline([
    ("sampler", SMOTENC( # Per prima cosa generiamo campioni sintetici sulle feature originali, rispettando le variabili categoriche
        categorical_features=categorical_indices, # serve a indicare a SMOTENC quali colonne (ricavate con l'index di queste) deve trattare come categoriche durante la generazione dei dati sintetici.
        # categorical_features e' la keyword del parametro previsto dall'API di SMOTENC
        random_state=42
    )),
    ("preprocessor", preprocessor_lr), # standardizza le numeriche e applica il one-hot encoding alle categoriche dopo la creazione dei dati sintetici creati da Smote
    ("classifier", LogisticRegression( # definisce il modello
        max_iter=1000,
        random_state=42
    ))
])

print(lr_smotenc)


# ----
# Passaggio successivo — Calcoliamo le probabilità OOF.

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

y_proba_oof_smotenc = cross_val_predict( # Usato per ottenere predict_proba o predict, con cross-validation oof
    clone(lr_smotenc), # pipeline che fa': oversampling, scaling, ohe. Fit solo sui dati di training nel fold, transform sul validation set.  (ps. oversampling solo su traning grups)
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

print("Numero di probabilità OOF:", len(y_proba_oof_smotenc))
print("Valori mancanti:", np.isnan(y_proba_oof_smotenc).sum())
print("Probabilità minima:", round(y_proba_oof_smotenc.min(), 4))
print("Probabilità massima:", round(y_proba_oof_smotenc.max(), 4))


# Confrontiamo i quattro modelli alla soglia 0.50

def valuta_oof(y_true, y_proba, soglia=0.50):
    y_pred = (y_proba >= soglia).astype(int)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "Brier score": brier_score_loss(y_true, y_proba)
    }

confronto_oof = pd.DataFrame({
    "Baseline": valuta_oof(y_train, y_proba_oof_lr),
    "Balanced": valuta_oof(y_train, y_proba_oof_balanced),
    "RandomOverSampler": valuta_oof(y_train, y_proba_oof_ros),
    "SMOTENC": valuta_oof(y_train, y_proba_oof_smotenc)
}).T

print(confronto_oof.round(4))

# precision leggermente superiore dei due modelli riequilibrati tramite ponderazione o duplicazione, ma il resto delle metriche euguale e leggermente piu' scarso

# ----
# selezioniamo la soglia OOF con le stesse caratteristiche che abbiamo adoperato per gli altri:

# Recall almeno pari a 0.70.
# Tra le soglie ammissibili, scegliamo quella con il minor numero di falsi positivi.
# Usiamo esclusivamente le predizioni OOF del training set per questa scelta.


risultati_smotenc = []

for soglia in np.arange(0.30, 0.701, 0.01):
    soglia = round(soglia, 2)
    y_pred = (y_proba_oof_smotenc >= soglia).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_train, y_pred, labels=[0, 1]
    ).ravel()

    risultati_smotenc.append({
        "Threshold": soglia,
        "Precision": precision_score(
            y_train, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_train, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_train, y_pred, zero_division=0
        ),
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "TN": tn
    })

tabella_smotenc = pd.DataFrame(risultati_smotenc)

ammissibili_smotenc = tabella_smotenc[
    tabella_smotenc["Recall"] >= 0.70
]

scelta_smotenc = ammissibili_smotenc.sort_values(
    ["FP", "Threshold"],
    ascending=[True, False]
).iloc[0]

print("Soglia selezionata secondo il criterio stabilito:")
print(scelta_smotenc.to_frame("Valore").round(4))

# la soglia selezionata è 0.53
# Alle soglie selezionate con il criterio stabilito, SMOTENC presenta un F1-score inferiore agli altri tre modelli e genera più 
# falsi positivi


# ------
# Fit finale e test set sul modello con sogli 0.53

lr_smotenc.fit(X_train, y_train)

print("Pipeline SMOTENC addestrata correttamente.")
print("Classi:", lr_smotenc.classes_)


# Probabilità di churn sul test set
y_proba_test_smotenc = lr_smotenc.predict_proba(X_test)[:, 1]

# Soglia selezionata tramite cross-validation sul training set
threshold_smotenc = 0.53

# Conversione delle probabilità in classi
y_pred_test_smotenc = (
    y_proba_test_smotenc >= threshold_smotenc
).astype(int)

# Metriche di valutazione
print(f"Threshold: {threshold_smotenc:.2f}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_test_smotenc):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test_smotenc):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_test_smotenc):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_test_smotenc):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba_test_smotenc):.4f}")

# Matrice di confusione: righe = valori reali, colonne = predizioni
tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred_test_smotenc,
    labels=[0, 1]
).ravel()

print("\nConfusion Matrix:")
print(f"TN: {tn} | FP: {fp}")
print(f"FN: {fn} | TP: {tp}")


# ----
# Confrontiamo tutti e quattro i modelli con rispettive soglie favorite

results_test = pd.DataFrame({
    "Baseline (0.36)": {
        "Accuracy": 0.7665,
        "Precision": 0.5470,
        "Recall": 0.7005,
        "F1-score": 0.6143,
        "ROC-AUC": 0.8420,
        "FP": 217,
        "FN": 112,
        "TP": 262,
        "TN": 818
    },
    "Balanced (0.60)": {
        "Accuracy": 0.7622,
        "Precision": 0.5399,
        "Recall": 0.7059,
        "F1-score": 0.6118,
        "ROC-AUC": 0.8416,
        "FP": 225,
        "FN": 110,
        "TP": 264,
        "TN": 810
    },
    "RandomOverSampler (0.61)": {
        "Accuracy": 0.7679,
        "Precision": 0.5495,
        "Recall": 0.6979,
        "F1-score": 0.6148,
        "ROC-AUC": 0.8421,
        "FP": 214,
        "FN": 113,
        "TP": 261,
        "TN": 821
    },
    "SMOTENC (0.53)": {
        "Accuracy": 0.7509,
        "Precision": 0.5230,
        "Recall": 0.6979,
        "F1-score": 0.5979,
        "ROC-AUC": 0.8210,
        "FP": 238,
        "FN": 113,
        "TP": 261,
        "TN": 797
    }
})

print(results_test.round(4))


# Il confronto ci permette già di osservare che:
# - RandomOverSampler ha ottenuto il miglior F1-score sul test set (0.6148), anche se il vantaggio rispetto al baseline (0.6143) è minimo.
# - Balanced Logistic Regression ha il recall più alto (0.7059), individuando 264 clienti churn su 374.
# - SMOTENC presenta i risultati meno convincenti, con F1-score 0.5979 e ROC-AUC 0.8210.

# La conclusione provvisoria è che le tecniche di bilanciamento non hanno prodotto un miglioramento sostanziale.
# In particolare, SMOTENC non sembra vantaggioso per questo modello e dataset.


# ------
# Tuning regolarizzazione L1 e L2 mantenendo inizialmente drop=None

# Controlliamo la configurazione attuale del OneHotEncoder
encoder = preprocessor_lr.named_transformers_["categorical"]

print("OneHotEncoder attuale:")
print(encoder)

print("\nParametro drop:")
print(encoder.drop)

print("\nCategorie gestite:")
print(encoder.categories_)


# Prepariamo il confronto

# definiamo le configurazioni


# Valori di C da confrontare
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

# Configurazioni della regressione logistica
regularization_configs = [
    {
        "penalty": penalty,
        "C": C  # determina l'intensita della regolazione. C basso = forte regolazione (underfitting). C alto il bassa regolazione (overfitting)
    }
    for penalty in ["l1", "l2"]
    for C in C_values
] # list comprehension

print(f"Numero di configurazioni: {len(regularization_configs)}")
print("\nConfigurazioni da testare:")

for config in regularization_configs:
    print(config)

#  cross-validation

# Memorizziamo le probabilità OOF per ogni configurazione (soglia standard di 0.5)
oof_regularization = {}

for config in regularization_configs: # creiamo una pipeline con value C dedicato per oni configurazione
    model = Pipeline([
        ("preprocessor", clone(preprocessor_lr)),
        ("classifier", LogisticRegression(
            penalty=config["penalty"], # Itera su L1, l2
            C=config["C"], # itera sui valori [0.01, 0.1, 1.0, 10.0, 100.0]
            solver="liblinear", # solver liblinear, compatibile con entrambe le penalizzazioni e con la classificazione binaria del nostro problema.
            max_iter=2000,
            random_state=42
        ))
    ])

    y_proba_oof = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv, # crea i fold, i gruppi train e validation, e si assicura che le classi di ogni fold siano partizionate correttamente (74% churn = no data, e 36 % churn = yes data)
        method="predict_proba", # method="predict_proba appartiene alla funxione cross_val_predict
        n_jobs=-1
    )[:, 1]

    key = f'{config["penalty"].upper()}_C={config["C"]}'
    oof_regularization[key] = y_proba_oof

    print(
        f"{key}: "
        f"{len(y_proba_oof)} probabilità, "
        f"NaN={np.isnan(y_proba_oof).sum()}, "
        f"min={y_proba_oof.min():.4f}, "
        f"max={y_proba_oof.max():.4f}"
    )


# ------
# confronto delle metriche OOF (soglia 0.5)


results_regularization = []

for name, y_proba in oof_regularization.items():
    y_pred = (y_proba >= 0.5).astype(int)

    results_regularization.append({
        "Modello": name,
        "Accuracy": accuracy_score(y_train, y_pred),
        "Precision": precision_score(y_train, y_pred, zero_division=0),
        "Recall": recall_score(y_train, y_pred, zero_division=0),
        "F1-score": f1_score(y_train, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_train, y_proba),
        "Brier score": brier_score_loss(y_train, y_proba)
    })

df_regularization = (
    pd.DataFrame(results_regularization)
    .sort_values("F1-score", ascending=False)
    .reset_index(drop=True)
)

print(df_regularization.round(4))

# Migliori configurazioni a soglia 0.5: L2_C=100.0 e L1_C=100.0, entrambe con F1-score 0.5978 e ROC-AUC 0.8459.

# L2 con C=0.01: ottiene la precision più alta (0.6690), ma il recall scende a 0.5164. Individua quindi una quota inferiore dei clienti che abbandonano.

# L1 con C=0.01: il risultato è nettamente peggiore, con F1-score 0.4161 e ROC-AUC 0.8249. Una penalizzazione L1 molto forte può azzerare troppi coefficienti utili.

# il modello non trae un grande beneficio da una penalizzazione più intensa.

# ---------
# confrontiamo le soglie 

# Per ogni configurazione cercheremo la soglia che soddisfa il criterio già utilizzato (>= 70% recall, con minori FP), senza consultare il test set.


threshold_results = []

for name, y_proba in oof_regularization.items():
    candidates = []

    # Valutiamo soglie candidate tra 0.10 e 0.90
    for threshold in np.arange(0.10, 0.901, 0.01):
        y_pred = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_train, y_pred, labels=[0, 1]
        ).ravel()

        recall = recall_score(y_train, y_pred, zero_division=0)

        if recall >= 0.70:
            candidates.append({
                "Soglia": threshold,
                "Precision": precision_score(
                    y_train, y_pred, zero_division=0
                ),
                "Recall": recall,
                "F1-score": f1_score(
                    y_train, y_pred, zero_division=0
                ),
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "TN": tn
            })

    if candidates:
        # Prima minimizziamo FP; a parità, preferiamo la soglia più alta
        best = sorted(
            candidates,
            key=lambda row: (row["FP"], -row["Soglia"])
        )[0]

        threshold_results.append({
            "Modello": name,
            **best
        })

df_threshold_regularization = (
    pd.DataFrame(threshold_results)
    .sort_values(["F1-score", "FP"], ascending=[False, True])
    .reset_index(drop=True)
)

print(df_threshold_regularization.round(4))

# La configurazione migliore secondo il criterio scelto è L2 con C=0.01 e soglia 0.36.


# ------
# addestriamo la configurazione selezionata


lr_l2_tuned = Pipeline([
    ("preprocessor", clone(preprocessor_lr)),
    ("classifier", LogisticRegression(
        penalty="l2",
        C=0.01,
        solver="liblinear",
        max_iter=2000,
        random_state=42
    ))
])

lr_l2_tuned.fit(X_train, y_train)

print("Modello L2 con C=0.01 addestrato correttamente.")
print("Classi:", lr_l2_tuned.classes_)
print("Soglia selezionata tramite OOF:", 0.36)


# Valutazione su test set:


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Probabilità di churn sul test set
y_proba_test_l2_tuned = lr_l2_tuned.predict_proba(X_test)[:, 1]

# Soglia scelta tramite cross-validation sul training set
threshold_l2_tuned = 0.36

# Classificazione
y_pred_test_l2_tuned = (
    y_proba_test_l2_tuned >= threshold_l2_tuned
).astype(int)

# Metriche
print(f"Threshold: {threshold_l2_tuned:.2f}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_test_l2_tuned):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test_l2_tuned):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_test_l2_tuned):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_test_l2_tuned):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba_test_l2_tuned):.4f}")

# Matrice di confusione
tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred_test_l2_tuned,
    labels=[0, 1]
).ravel()

print("\nConfusion Matrix:")
print(f"TN: {tn} | FP: {fp}")
print(f"FN: {fn} | TP: {tp}")

# Le differenze sono minime tra la baseline e il modello dove abbiamo applicato il tuning dell regolarizazzione 
# La regolarizzazione più forte non ha apportato un vantaggio concreto nel nostro esperimento. 

# -------
# -------

# Feature selection

# Misuriamo la correlazione tra le variabili indipendenti, e' vediamo se si riesce a migliorare le metriche allenando il modello senza feature correlate

cols = ["tenure", "TotalCharges"]

print(df_analysis[cols].corr(method="pearson")) # Intensità della relazione lineare
print(df_analysis[cols].corr(method="spearman")) # Relazione monotona, anche non lineare

# forte relazione tra queste 2 features

# visualiziamo la relazione:

plt.scatter(df_analysis["tenure"], df_analysis["TotalCharges"], alpha=0.3)
plt.xlabel("Tenure")
plt.ylabel("TotalCharges")
plt.title("Tenure vs TotalCharges")
plt.show()

# Evidente relazione

# Correlazioni tra tutte le feature numeriche
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

print("Correlazione Pearson:")
print(df_analysis[numeric_cols].corr(method="pearson").round(3))

print("Correlazione Spearman:")
print(df_analysis[numeric_cols].corr(method="spearman").round(3))

# MonthlyCharges – TotalCharges -> 0,651 -- 0,638	Correlazione moderata/forte 

# ----
# associazioni tra feature categoriche

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from itertools import combinations

# Feature categoriche, escludendo il target Churn
cat_cols = categorical_features

cramers_v = pd.DataFrame(
    np.eye(len(cat_cols)),
    index=cat_cols,
    columns=cat_cols
)

for col1, col2 in combinations(cat_cols, 2):
    contingency = pd.crosstab(df_analysis[col1], df_analysis[col2])
    chi2 = chi2_contingency(contingency, correction=False)[0]
    n = contingency.to_numpy().sum()
    r, k = contingency.shape

    # V di Cramér con correzione del bias
    phi2 = chi2 / n
    phi2_corr = max(
        0,
        phi2 - ((k - 1) * (r - 1)) / (n - 1)
    )
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)

    denominator = min(k_corr - 1, r_corr - 1)
    v = np.sqrt(phi2_corr / denominator) if denominator > 0 else np.nan

    cramers_v.loc[col1, col2] = v
    cramers_v.loc[col2, col1] = v

print(cramers_v.round(3))

# Coppia di feature	------ ----  V di Cramér	Interpretazione
# PhoneService – MultipleLines - 1,000	------- Associazione perfetta secondo questa misura
# Molte features categoriche presentano una correlazione forte


# ------

# Verifichiamo PhoneService e MultipleLines

print(pd.crosstab(
    df_analysis["PhoneService"],
    df_analysis["MultipleLines"],
    margins=True
))

print("\nValori di MultipleLines per ciascun valore di PhoneService:")
for value in df_analysis["PhoneService"].unique():
    print(f"\nPhoneService = {value}")
    print(
        df_analysis.loc[
            df_analysis["PhoneService"] == value,
            "MultipleLines"
        ].value_counts(dropna=False)
    )

# feature perfettamente associate nel dataset.

# ---

# Completiamo l'analisi delle associazioni tra feature numeriche e categoriche



numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = categorical_features

def correlation_ratio(categories, values):
    data = pd.DataFrame({
        "category": categories,
        "value": values
    }).dropna()

    if data["category"].nunique() < 2:
        return np.nan

    grand_mean = data["value"].mean()
    ss_between = sum(
        len(group) * (group.mean() - grand_mean) ** 2
        for _, group in data.groupby("category")["value"]
    )
    ss_total = ((data["value"] - grand_mean) ** 2).sum()

    return np.sqrt(ss_between / ss_total) if ss_total > 0 else np.nan

eta_matrix = pd.DataFrame(
    index=cat_cols,
    columns=numeric_cols,
    dtype=float
)

for cat in cat_cols:
    for num in numeric_cols:
        eta_matrix.loc[cat, num] = correlation_ratio(
            df_analysis[cat],
            df_analysis[num]
        )

print(eta_matrix.round(3))


# ---------
# Confrontare modelli alternativi facendo feature selection sulle features correlate secondo i risultati delle analisi che abbiamo appena completato per individuare le coppie e i gruppi candidati.

# Confrontiamo le prestazioni in cross-validation di modelli che differiscono per una sola scelta di feature. 
# È il modo più diretto per capire se una variabile può essere rimossa senza compromettere il risultato.

# ----
# Iniziamo con la possibile eliminazione di PhoneService, perche' MultipleLines contiene anche le informazioni sull'assenza del servizio telefonico

# Modello A: tutte le feature
X_train_all = X_train.copy()

# Modello B: rimuoviamo solo PhoneService
X_train_no_phone = X_train.drop(columns=["PhoneService"]).copy()

# Ricreiamo il preprocessore per il modello ridotto
categorical_features_no_phone = [
    col for col in categorical_features if col != "PhoneService"
]

print(categorical_features_no_phone)

# ColumnTransformer dedicato, senza PhoneService column
preprocessor_no_phone = ColumnTransformer(transformers=[
    ("numeric", StandardScaler(), numeric_features),
    ("binary", "passthrough", binary_features),
    ("categorical",
     OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     categorical_features_no_phone)
])

model_all = clone(lr_baseline) # preprocessor e classifier

model_no_phone = Pipeline([ # stessa pipeline di lr_baseline, senza la colonna PhoneService
    ("preprocessor", preprocessor_no_phone),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

print("Feature modello A:", X_train_all.shape[1])
print("Feature modello B:", X_train_no_phone.shape[1])
print("PhoneService presente nel modello B:",
      "PhoneService" in X_train_no_phone.columns)


# Cross-validation con predizioni OOF

# Probabilità OOF del modello completo
# già calcolate prima per la Logistic Regression baseline
y_proba_oof_all = y_proba_oof_lr

# Probabilità OOF del modello senza PhoneService
y_proba_oof_no_phone = cross_val_predict(
    clone(model_no_phone),
    X_train_no_phone,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

# Metriche con soglia standard 0.50
def evaluate_oof(y_true, y_proba, threshold=0.50):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba)
    }

results_phone = pd.DataFrame({
    "Modello completo": evaluate_oof(y_train, y_proba_oof_all),
    "Senza PhoneService": evaluate_oof(y_train, y_proba_oof_no_phone)
}).T

print(results_phone.round(4))

# Le differenze sulle metriche sono estremamente piccole. Questo è coerente con quanto avevamo osservato: MultipleLines contiene già l'informazione relativa all'assenza del servizio telefonico.
# Possiamo considerare PhoneService una candidata alla rimozione.


# ----

# Valutiamo tenure e TotalCharges:

# Confronteremo:
# - Modello A: tutte le feature, come riferimento.
# - Modello B: senza tenure.
# - Modello C: senza TotalCharges.


# Configurazione A: tutte le feature


# Configurazione B: senza tenure
X_train_no_tenure = X_train.drop(columns=["tenure"]).copy() 

# Configurazione C: senza TotalCharges
X_train_no_totalcharges = X_train.drop(columns=["TotalCharges"]).copy()


def build_preprocessor(X):
    numeric_cols = [c for c in numeric_features if c in X.columns]
    binary_cols = [c for c in binary_features if c in X.columns]
    categorical_cols = [c for c in categorical_features if c in X.columns]

    return ColumnTransformer(transformers=[
        ("numeric", StandardScaler(), numeric_cols),
        ("binary", "passthrough", binary_cols),
        ("categorical",
         OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical_cols)
    ])


model_no_tenure = Pipeline([
    ("preprocessor", build_preprocessor(X_train_no_tenure)),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

model_no_totalcharges = Pipeline([
    ("preprocessor", build_preprocessor(X_train_no_totalcharges)),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

print("Feature modello A:", X_train_all.shape[1])
print("Feature modello B:", X_train_no_tenure.shape[1])
print("Feature modello C:", X_train_no_totalcharges.shape[1])


# ---
# Calcoliamo le probabilità OOF e confrontiamo le metriche


# Modello A: riutilizziamo le probabilità OOF già calcolate
y_proba_oof_all = y_proba_oof_lr

# Modello B: senza tenure
y_proba_oof_no_tenure = cross_val_predict(
    clone(model_no_tenure),
    X_train_no_tenure,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

# Modello C: senza TotalCharges
y_proba_oof_no_totalcharges = cross_val_predict(
    clone(model_no_totalcharges),
    X_train_no_totalcharges,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]


# Valutazione con soglia standard 0.50
results_numeric_features = pd.DataFrame({
    "Completo": evaluate_oof(y_train, y_proba_oof_all, threshold=0.50),
    "Senza tenure": evaluate_oof(y_train, y_proba_oof_no_tenure, threshold=0.50),
    "Senza TotalCharges": evaluate_oof(y_train, y_proba_oof_no_totalcharges, threshold=0.50)
}).T

print(results_numeric_features.round(4))

# Le metriche suggeriscono che tenure fornisce informazione predittiva utile che non viene completamente recuperata dalle altre feature.
# Decisione provvisoria: Tenure da mantenere, mentre TotalCharges candidata alla rimozione

# ----
# Prossimo passo:
# Confronto tra modello completo e quello senza TotalCharges usando la procedura già adottata nel progetto: 
# 1) Selezionare per ciascun modello una soglia OOF che raggiunga recall almeno pari a 0,70, minimizzando i falsi positivi.
# 2) Confrontare precision, recall, F1-score e matrice di confusione alle rispettive soglie.
# 3) Solo dopo, verificare la variante scelta sul test set.

# -----

# Selezione della soglia per entrambi i modelli   


def select_threshold_by_recall(y_true, y_proba, min_recall=0.70):
    results = []

    # Valutiamo le soglie distinte presenti nelle probabilità OOF
    for threshold in np.sort(np.unique(y_proba))[::-1]: # le soglie sulla quale iteriamo equivalgono hai probs dati oof del modello sistemati in ordine ascendente
        y_pred = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        recall = recall_score(y_true, y_pred, zero_division=0)

        if recall >= min_recall:
            results.append({
                "Soglia": threshold,
                "Precision": precision_score(
                    y_true, y_pred, zero_division=0
                ),
                "Recall": recall,
                "F1-score": f1_score(
                    y_true, y_pred, zero_division=0
                ),
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "TN": tn
            })

    if not results:
        return None

    # Prima minimizziamo i falsi positivi; a parità, preferiamo
    # la soglia più alta.
    return (
        pd.DataFrame(results)
        .sort_values(
            ["FP", "Soglia"],
            ascending=[True, False]
        )
        .iloc[0] # ritorna le soglia e i le metriche, a queste condizioni: Recal >= 70 e min FP
    )

threshold_full = select_threshold_by_recall(
    y_train, y_proba_oof_all, min_recall=0.70
) # # ritorna le soglia e i le metriche, a queste condizioni: Recal >= 70 e min FP

threshold_no_total = select_threshold_by_recall(
    y_train, y_proba_oof_no_totalcharges, min_recall=0.70
) # # ritorna le soglia e i le metriche, a queste condizioni: Recal >= 70 e min FP

results_threshold = pd.DataFrame({
    "Modello completo": threshold_full,
    "Senza TotalCharges": threshold_no_total
}).T

print(results_threshold.round(4).to_string())

# Il modello senza TotalCharges raggiunge lo stesso recall e lo stesso numero di falsi negativi, 
# ma presenta leggermente meno falsi positivi.

# ----
# Verifica sul test set


# Fit del modello "no_totalcharges" sul training set completo (gia fatto per lr_baseline)

model_no_total_fitted = clone(model_no_totalcharges)
model_no_total_fitted.fit(X_train_no_totalcharges, y_train)

# Probabilità sul test set del modello baseline
proba_test_all = y_proba_lr # Probs calcolate in precedenza dopo fit su X_train e X_test, e probs su X_test[:, 1] 

# Probabilità sul test set del modello baseline senza colonna TotalCharges
proba_test_no_total = model_no_total_fitted.predict_proba(
    X_test.drop(columns=["TotalCharges"])
)[:, 1]

# Soglie scelte esclusivamente tramite OOF sul training set
threshold_all = float(threshold_full["Soglia"]) # 0.363..
threshold_no_total = float(threshold_no_total["Soglia"])

def evaluate_test_model(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "Soglia": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "TN": tn
    }

test_comparison = pd.DataFrame({
    "Modello completo": evaluate_test_model(
        y_test, proba_test_all, threshold_all
    ),
    "Senza TotalCharges": evaluate_test_model(
        y_test, proba_test_no_total, threshold_no_total
    )
}).T

print(test_comparison.round(4).to_string())

# Prestazioni sono molto simili
# Il modello senza TotalCharges segnala erroneamente tre clienti in meno come potenziali abbandoni, ma ne perde tre in più tra quelli che abbandonano realmente.


# ----------
# Selezione delle soglie OOF tra y_proba_oof_all e model_no_phone per vedere se le prestazioni cambiano trovando una soglia con >= 70 % recall e meno FP possibili.

# Usiamo "y_proba_oof_all" e il modello "model_no_phone" per confrontare i due modelli con il vincolo Recall >= 0.70


# Probabilità OOF del modello completo, già calcolate prima: (y_proba_oof_all = y_proba_oof_lr)
# y_proba_oof_all = y_proba_oof_lr

# Probabilità OOF del modello senza PhoneService Gia calcolate prima: (y_proba_oof_no_phone)

# Selezione della soglia per ciascun modello
threshold_full_phone = select_threshold_by_recall(
    y_train,
    y_proba_oof_all,
    min_recall=0.70
)

threshold_no_phone = select_threshold_by_recall(
    y_train,
    y_proba_oof_no_phone,
    min_recall=0.70
)

# Tabella di confronto
results_phone_threshold = pd.DataFrame({
    "Modello completo": threshold_full_phone,
    "Senza PhoneService": threshold_no_phone
}).T

print(results_phone_threshold.round(4).to_string())

# Nessun beneficio tangibile.
# sembra che la rimozione non porta vantaggi predittivi.


# -----
# Verifichiamo anche il test set


model_no_phone_fitted = clone(model_no_phone)
model_no_phone_fitted.fit(X_train_no_phone, y_train)

# Probabilità sul test set baseline, (gia fitted e proba eseguite su X_test)
proba_test_all_phone = y_proba_lr

proba_test_no_phone = model_no_phone_fitted.predict_proba(
    X_test.drop(columns=["PhoneService"]) # droppiamo PhoneService dal test set
)[:, 1]

# Soglie selezionate sul training set tramite OOF
threshold_all_phone = float(threshold_full_phone["Soglia"])
threshold_reduced_phone = float(threshold_no_phone["Soglia"])

# Confronto sul test set
results_test_phone = pd.DataFrame({
    "Modello completo": evaluate_test_model(
        y_test, proba_test_all_phone, threshold_all_phone
    ),
    "Senza PhoneService": evaluate_test_model(
        y_test, proba_test_no_phone, threshold_reduced_phone
    )
}).T

print(results_test_phone.round(4).to_string())

# Neanche qui abbiamo avuto nessun cambiamento sostanziale analizzando e applicando feature selection sulle features: tenure, TotalCharges, MultipleLines e PhoneService
# In questo esperimento abbiamo constatato che non c'e' nessun beneficio tangibile nella rimozione di queste features.

# Quindi per il momento manteniamo tutte le feature.

# -----

# Intuitivmente, applichiamo la stessa logica applicata fino ad ora per altre features che risultano correlate tra loro al con intensita' >= 70%, ad esempio StreamingTV e StreamingMovies

# ............


