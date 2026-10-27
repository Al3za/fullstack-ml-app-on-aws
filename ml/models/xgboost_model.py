# TREE_MODELS

import pandas as pd
import matplotlib.pyplot as plt
from dataset import df_analysis, X, y, numeric_features, binary_features, categorical_features
from preprocessing import make_categorical_encoder, make_preprocessor
from data_split import X_train, X_test, y_train, y_test

# Intro dei dati preprocessati importati

print("=== CHECK IMPORTED X,y DATA ===")

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
print("=== CHECK IMPORTED FEATURE GROUPS ===")

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

# Encoding del target

print("=== CHECK IMPORTED TARGET DATA ENCODED ===")

print("\n--- Target dopo la conversione ---")
print(y.value_counts())

print("\n--- Target dtype ---")
print(y.dtype)

print("\n--- Valori mancanti ---")
print(y.isna().sum())

# ---------

# Crea Encoding delle feature categoriche.

from sklearn.preprocessing import OneHotEncoder

# categorical_encoder = make_categorical_encoder() # preprocessing.py import

# print(categorical_encoder)

# categorical_encoder viene applicato nel file preprocessor.py. (lo inseriamo qui' solo a scopo illustrativo)

# --------------

# Costruire il ColumnTransformer. Fondamentale per la Pipeline che andremo a creare

from sklearn.compose import ColumnTransformer

preprocessor = make_preprocessor(numeric_features, binary_features, categorical_features) # preprocessing.py import

print(preprocessor)

print("=== CHECK PREPROCESSOR  ===")
print("\n--- Feature totali ---")
print(len(numeric_features) + len(binary_features) + len(categorical_features)) # 19

#

print("\n--- Preprocessor configurato ---")
print(preprocessor.transformers) ## "mostra la configurazione dei trasformatori che ho definito."


# --------
# Controlla Train/Test Split e Fit e trasformazione delle feature

from sklearn.model_selection import train_test_split

print("CHECK IMPORTED TRAIN / TEST SPLIT ===")

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
# Eseguiamo il fit e la trasformazione dei dati per verificare che il preprocessor funziona e le trasformazioni OHE avvengono correttamente
# Essenziale da verificare prima di usare la pipeline nel Cross-validation

print("=== FIT & TRANSFORM di train e test dat ===")

X_train_processed = preprocessor.fit_transform(X_train) # one hot encoder applied in preprocessing.py 
X_test_processed = preprocessor.transform(X_test)


print("\n--- Shapes dopo preprocessing ---")
print("X_train_processed:", X_train_processed.shape)
print("X_test_processed :", X_test_processed.shape)
# 45 colonne delle x features dopo transformazione encoding

# Trasformiamo l'array numpy in dataframe pandas solo per mostrare la trasformazione dell encoder sulle variabili categortiche¶
feature_names = preprocessor.get_feature_names_out() # get column

X_train_processed_df = pd.DataFrame( # transform numpy array in Pandas datafraeme
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

X_train_processed_df
# ------

# Verifica del preprocessing

print("=== PROCESSED DATA CHECK ===")

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


# -----
# -----

# PIPELINE, CROSS-VALIDATION e OOF
# Creiamo la pipeline dove inseriamo il ColumnTransformer creato in precedenza, in modo da non avere
# Data leakage durante il cross-validation k-fold

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
) 

preprocessor = make_preprocessor(numeric_features, binary_features, categorical_features)


from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from scipy.stats import randint # permette di definire intervalli di valori interi da esplorare,
# anziché elencare ogni valore possibile.


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



# Preparare la pipeline XGBoost

# Ricordiamo che La pipeline applica il preprocessing e poi addestra XGBoost

from sklearn.base import clone # non modifichiamo ripetutamente la pipeline, creiamo invece una nuova pipeline per ogni configurazione, riutilizzando il preprocessor.
# l'import clone(preprocessor), fa' in modo che il preprocessor sia riutilizzabile, facendo 'dimenticare' al preprocessor i precedenti fit.transform del train set. 
# Questo evita data leakage


xgb_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", XGBClassifier(
        objective="binary:logistic", #→ classificazione binaria(yes - no) con output probabilistico(per misurare auc score)
        eval_metric="logloss", # metrica coerente con una classificazione binaria probabilistica
        random_state=42,
        n_jobs=1
    ))
])

print("Pipeline XGBoost creata correttamente.")
print(xgb_pipeline)


# -----------
# Definiamo la ricerca degli iperparametri con RandomizedSearchCV, e poi facciamo un piccolo tuning manualmente con cross-validate su alcuni iperparametri
# Usiamo queste strategie per scegliere gli iperparametri perche' cosi' valutiamo le metriche direttamente sul train set, lasciando il test set "indipendente" per il test finale.


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


xgb_search = RandomizedSearchCV(
    estimator=xgb_pipeline, # trasformiamo le colonne categoriche con OneHotEncoder e applichiamo il modello per il traning
    param_distributions=xgb_param_distributions, # gli intervalli degli iperparametri che vogliamo analizzare per ottenere il miglior f1 score, come stabilito in "scoring"
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
# Ulteriore tuning "manuale" su XGBoost di alcuni parametri, lasciando fissi quelli ottenuti con RandomizedSearchCV:
# Il tuning verra' sempre fatto con cross-validate

# min_child_weight: 

# Parametri migliori trovati da RandomizedSearchCV
best_params = xgb_search.best_params_.copy()

# Copiamo i parametri ottenti da RandomizedSearchCV per poi copiarli nella cross-validate.
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

# min_child_weight=1 rimane la configurazione migliore, quindi non andremo a cambiare niente dagli iperparametri trovati da RandomizeSearrchCVs

# -------
# -------

# Consolidiamo e addestriamo il modello finale

# Utilizziamo direttamente xgb*search.best_estimator*, che contiene già la pipeline con il preprocessing,
# gli parametri mifgliori per f1 score, ed è già stato riaddestrato su tutto X_train grazie a refit=True.

# Modello XGBoost consolidato grazie a refit=True.
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

# metriche:

# METRICHE XGBOOST
# Metrica Valore
# Accuracy 0.8048
# Precision 0.6667
# Recall 0.5294
# F1-score 0.5902
# ROC-AUC 0.8472

# CONFUSION MATRIX
# [[936  99]
#  [176 198]]

# XGBoost vince ancora e si candida come modello favorito sopra Random Forest. Semplicemente perche' raggiunge un F1 più alto e una
# precision maggiore

# -----
# -----

# Find Best threshold:

# Cerchiamo la soglia decisionale ideale per aumentare i TP al minor costo possibile di FP.

# Scegliamo la soglia usando le predizioni out-of-fold

# Generiamo le probabilità out-of-fold


from sklearn.model_selection import cross_val_predict


# Ogni osservazione del training set viene predetta
# da un modello che non ha visto quell'osservazione
oof_proba = cross_val_predict(
    clone(xgb_final), # clone() crea una copia non addestrata del modello, mantenendo gli iperparametri selezionati. # cross_val_predict avvia addestramento del modello vergine clonato sui fold previsti dalla cross-validation.
    X_train,
    y_train,
    cv=cv,
    method="predict_proba", # Valutazione con predizioni OOF
    n_jobs=-1
)[:, 1] # ritorna le probs dei churn = 1

print("Predizioni out-of-fold generate:", len(oof_proba)) # lunghezza dei dati di training. Corrisponde con in nr di rows del traning set


# ----
# Tuning delle le soglie. Vediamo quale di queste soglie offre miglior trade-off tra precision e recall

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
# ------

# Applichiamo la soglia candidata al test set:

# Condizioni: Recall almeno del 70%, cercando di contenere gli FP

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

# Soglia selezionata tramite probs OOF
threshold = 0.35
y_test_pred_035 = (y_test_proba >= threshold).astype(int) # True=1

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

# es:

# scale_pos_weights = [1.0, 1.5, 2.0, 2.8]

# weight_results = []

# for weight in scale_pos_weights:
#     model = clone(xgb_search.best_estimator_)

#     model.set_params(
#         classifier__scale_pos_weight=weight
#     )

#     scores = cross_validate(
#         model,
#         X_train,
#         y_train,
#         cv=cv,
#         scoring={
#             "precision": "precision",
#             "recall": "recall",
#             "f1": "f1",
#             "roc_auc": "roc_auc"
#         },
#         n_jobs=-1
#     )

#     weight_results.append({
#         "scale_pos_weight": weight,
#         "Precision_mean": scores["test_precision"].mean(),
#         "Recall_mean": scores["test_recall"].mean(),
#         "F1_mean": scores["test_f1"].mean(),
#         "ROC_AUC_mean": scores["test_roc_auc"].mean(),
#         "F1_std": scores["test_f1"].std()
#     })

# weight_results_df = pd.DataFrame(weight_results)

# print( 
#     weight_results_df
#     .sort_values("F1_mean", ascending=False)
#     .round(4)
#     .to_string(index=False)
# )


# misura le metriche....

# -------
# -------

###  Permutation Importance Individuale. 

# Iniziamo valutando le feature singolarmente. 
# Successivamente, raggrupperemo quelle che risultano associate, seguendo i criteri definiti nel file inspect_dataset.py.

# Per la nostra Permutation Importance, possiamo usare il modello xgb_final e la soglia 0.35. Il moodello finale con miglior tradeoff tra precision e recall (miglior f1)

# **Passaggio 1 — Calcoliamo la Permutation Importance**

from sklearn.inspection import permutation_importance


# Soglia selezionata tramite OOF validation (la soglia migliore per il nostro modello finale che porta al miglior f1 score)
threshold = 0.35

# Funzione per calcolare l'F1 usando la soglia personalizzata. 
# Per ogni feature, mescoleremo i valori 20 volte e misuriamo quanto cambia l'F1-score.

def f1_at_threshold(estimator, X, y):
    probabilities = estimator.predict_proba(X)[:, 1] # probs di churn=1 del modello candidato xgb_final 
    predictions = (probabilities >= threshold).astype(int)
    return f1_score(y, predictions)

# Permutation Importance sul modello XGBoost tuned
perm_importance_xgb = permutation_importance(
    estimator=xgb_final,
    X=X_test,
    y=y_test,
    scoring=f1_at_threshold,
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

print("Permutation Importance calcolata correttamente.")
print("Numero di feature analizzate:", len(perm_importance_xgb.importances_mean))


# Passaggio 2 — Visualizziamo le feature più importanti

# Creiamo una tabella con i risultati
perm_importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance_Mean": perm_importance_xgb.importances_mean, # media dei valori mescolati 20 volte per features
    "Importance_Std": perm_importance_xgb.importances_std # devisazione standard
})

# Ordiniamo dalla più importante alla meno importante
perm_importance_df = perm_importance_df.sort_values(
    by="Importance_Mean",
    ascending=False
).reset_index(drop=True)

# Stampiamo la classifica completa
# display(perm_importance_df.round(4))
print(perm_importance_df.round(4))

# Secondo questa prima analisi individuale, Contract e' nettamente la features piu' importante per la previsione del target.
# Riguardo alle features associate, risulta invece che tenure e InternetService sono le piu' importanti


# Visualizziamo le prime 15 feature
top_15 = perm_importance_df.head(15).sort_values(
    by="Importance_Mean",
    ascending=True
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_15["Feature"],
    top_15["Importance_Mean"],
    xerr=top_15["Importance_Std"]
)
plt.axvline(x=0, linestyle="--")
plt.xlabel("Diminuzione media dell'F1-score")
plt.ylabel("Feature")
plt.title("Permutation Importance — XGBoost tuned (soglia 0.35)")
plt.tight_layout()
plt.show()


# ----
# ----

# Permutazione per gruppi. Ragruppiamo le features associate calcolate nel file inspect_dataset.py, e raggruppiamole per misurare nuovamente la Permutation_importance


# Analiziamo il primo gruppo di features associate:

# - tenure
# - TotalCharges


import numpy as np
from sklearn.metrics import f1_score

# Gruppo di feature associate
group_features = ["tenure", "TotalCharges"]

# F1 di riferimento, senza permutazioni
baseline_f1 = f1_at_threshold(xgb_final, X_test, y_test)

group_importances = []

rng = np.random.default_rng(42)

for _ in range(20):
    X_permuted = X_test.copy()

    # Stessa permutazione delle righe per entrambe le feature:
    # la relazione interna tra tenure e TotalCharges viene preservata
    permutation = rng.permutation(len(X_permuted))

    X_permuted[group_features] = (
        X_test[group_features].iloc[permutation].to_numpy()
    )

    permuted_f1 = f1_at_threshold(
        xgb_final,
        X_permuted,
        y_test
    )

    # Quanto peggiora l'F1 rispetto al riferimento
    group_importances.append(baseline_f1 - permuted_f1)

group_importance_mean = np.mean(group_importances)
group_importance_std = np.std(group_importances, ddof=0)

print(f"F1-score di riferimento: {baseline_f1:.4f}")
print(f"Importanza congiunta media: {group_importance_mean:.4f}")
print(f"Deviazione standard: {group_importance_std:.4f}")

# output:

# F1-score di riferimento: 0.6318
# Importanza congiunta media: 0.1152
# Deviazione standard: 0.0155

# La permutazione congiunta di tenure e TotalCharges provoca una diminuzione dell'F1-score considerevole.
# Questo indica che il gruppo contiene informazioni predittive importanti per il tuo modello.

# -----
# -----

# Analizzare il secondo gruppo di features associate: 

# - PhoneService
# - MultipleLines

# Gruppo di feature associate
group_features = ["PhoneService", "MultipleLines"]

# F1 di riferimento
baseline_f1 = f1_at_threshold(xgb_final, X_test, y_test)

group_importances_phone = []

rng = np.random.default_rng(42)

for _ in range(20):
    X_permuted = X_test.copy()

    # Stessa permutazione per entrambe le feature
    permutation = rng.permutation(len(X_permuted))

    X_permuted[group_features] = (
        X_test[group_features].iloc[permutation].to_numpy()
    )

    permuted_f1 = f1_at_threshold(
        xgb_final,
        X_permuted,
        y_test
    )

    group_importances_phone.append(
        baseline_f1 - permuted_f1
    )

group_importance_mean_phone = np.mean(group_importances_phone)
group_importance_std_phone = np.std(
    group_importances_phone,
    ddof=0
)

print(f"F1-score di riferimento: {baseline_f1:.4f}")
print(
    "Importanza congiunta media: "
    f"{group_importance_mean_phone:.4f}"
)
print(
    "Deviazione standard: "
    f"{group_importance_std_phone:.4f}"
)

# F1-score di riferimento: 0.6318
# Importanza congiunta media: 0.0058 -> bassa
# Deviazione standard: 0.0044

# PhoneService e MultipleLines mostrano un'importanza congiunta piuttosto bassa, nonostante avere un'associazione forte secondo il calcolo
# Chi-square-test e cramer


# ----

# Analisi del terzo gruppo di features associate:

# Gruppo di servizi Internet e servizi aggiuntivi
group_features_internet = [
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

# F1 di riferimento
baseline_f1 = f1_at_threshold(xgb_final, X_test, y_test)

group_importances_internet = []

rng = np.random.default_rng(42)

for _ in range(20):
    X_permuted = X_test.copy()

    # Applichiamo la stessa permutazione a tutte le feature del gruppo
    permutation = rng.permutation(len(X_permuted))

    X_permuted[group_features_internet] = (
        X_test[group_features_internet]
        .iloc[permutation]
        .to_numpy()
    )

    permuted_f1 = f1_at_threshold(
        xgb_final,
        X_permuted,
        y_test
    )

    group_importances_internet.append(
        baseline_f1 - permuted_f1
    )

group_importance_mean_internet = np.mean(
    group_importances_internet
)
group_importance_std_internet = np.std(
    group_importances_internet,
    ddof=0
)

print(f"F1-score di riferimento: {baseline_f1:.4f}")
print(
    "Importanza congiunta media: "
    f"{group_importance_mean_internet:.4f}"
)
print(
    "Deviazione standard: "
    f"{group_importance_std_internet:.4f}"
)


# Il terzo gruppo ha un’importanza congiunta media di 0.1127, quindi la permutazione simultanea dei servizi Internet provoca un calo 
# significativo dell’F1-score. 

# Cio' indica che il modello perde una parte importante della sua capacità predittiva quando gli togliamo l'associazione originale
# tra i clienti e il loro gruppo di servizi. 

# ----------
# ----------

# Analisi del quarto gruppo di features associate:

# Contract + tenure

# Gruppo: contratto e durata del cliente
group_features_contract = ["Contract", "tenure"]

# F1 di riferimento
baseline_f1 = f1_at_threshold(xgb_final, X_test, y_test)

group_importances_contract = []
rng = np.random.default_rng(42)

for _ in range(20):
    X_permuted = X_test.copy()

    # Stessa permutazione per entrambe le feature
    permutation = rng.permutation(len(X_permuted))

    X_permuted[group_features_contract] = (
        X_test[group_features_contract]
        .iloc[permutation]
        .to_numpy()
    )

    permuted_f1 = f1_at_threshold(
        xgb_final,
        X_permuted,
        y_test
    )

    group_importances_contract.append(
        baseline_f1 - permuted_f1
    )

group_importance_mean_contract = np.mean(
    group_importances_contract
)
group_importance_std_contract = np.std(
    group_importances_contract,
    ddof=0
)

print(f"F1-score di riferimento: {baseline_f1:.4f}")
print(
    "Importanza congiunta media: "
    f"{group_importance_mean_contract:.4f}"
)
print(
    "Deviazione standard: "
    f"{group_importance_std_contract:.4f}"
)

# Questo è il risultato più alto tra i gruppi che abbiamo analizzato finora
# Il modello passa da un F1-score di 0.6318 a circa 0.4162 quando permutiamo insieme Contract e tenure. 
# Quindi il gruppo contiene informazioni predittive molto importanti per il modello.