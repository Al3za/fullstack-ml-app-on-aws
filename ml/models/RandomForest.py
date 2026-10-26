#  Random Forest

import pandas as pd
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


# TASK 2 — Pulizia e tipizzazione delle feature
# Partiamo da X, non da df_analysis, perché abbiamo già separato target e identificatore.

print("\n--- Data types prima della conversione ---")
print(X.dtypes)

print("\n--- Missing values prima della conversione ---")
print(X.isna().sum().sort_values(ascending=False))


# ---------
# TASK 3 — Separazione delle feature numeriche e categoriche
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

# TASK 4 — Encoding del target

print("=== CHECK IMPORTED TARGET DATA ENCODED ===")

print("\n--- Target dopo la conversione ---")
print(y.value_counts())

print("\n--- Target dtype ---")
print(y.dtype)

print("\n--- Valori mancanti ---")
print(y.isna().sum())

# ---------

# TASK 5 — Encoding delle feature categoriche

from sklearn.preprocessing import OneHotEncoder

# categorical_encoder = make_categorical_encoder() # preprocessing.py import

# print(categorical_encoder)

# categorical_encoder viene applicato nel file preprocessor.py. (lo inseriamo qui' solo a scopo illustrativo)

# --------------

# TASK 6 — Costruire il ColumnTransformer. Fondamentale per la Pipeline che andremo a creare

from sklearn.compose import ColumnTransformer

preprocessor = make_preprocessor(numeric_features, binary_features, categorical_features)

print(preprocessor)

print("=== CHECK PREPROCESSOR  ===")
print("\n--- Feature totali ---")
print(len(numeric_features) + len(binary_features) + len(categorical_features)) # 19

#

print("\n--- Preprocessor configurato ---")
print(preprocessor.transformers) ## "mostra la configurazione dei trasformatori che ho definito."


# --------
# TASK 7 — Controlla Train/Test Split e Fit e trasformazione delle feature

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
# Eseguiamo il fit e la trasformazione

print("=== FIT & TRANSFORM di train e test dat ===")

X_train_processed = preprocessor.fit_transform(X_train) # one hot encoder applied 
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

# TASK 8 — Verifica del preprocessing

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

# --------
# --------

# PIPELINE e CROSS-VALIDATION

# Creiamo una pipeline e usiamo la cross-validation sul training set per selezionare i parametri, lasciando il test set indipendente per la valutazione finale.

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
) 


preprocessor = make_preprocessor(numeric_features, binary_features, categorical_features)

rf_pipeline = Pipeline([
    ("preprocessor",preprocessor),
    ("classifier",RandomForestClassifier(random_state=42)),
])

print(rf_pipeline) # pipeline creata

# -------
# Preparare la ricerca degli iperparametri per Random Forest

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from scipy.stats import randint # permette di definire intervalli di valori interi da esplorare,
# anziché elencare ogni valore possibile.

# -----
# Preparare la cross-validation

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
# StratifiedKFold mantiene approssimativamente la stessa proporzione di clienti churn e non-churn in ogni fold.(73% / 26%) Si usa sempre con un dataset "sbilanciato"

# -------
# 4. Definire lo spazio di ricerca degli iperparametri per RandomizedSearchCV

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
    estimator=rf_pipeline, # Per ogni fold, estimator applica "fit.transform" del preprocessing definito nella pipeline solo sul train set, e poi esegue solo "transform" sul validation set del rispettivo fold. Questo Evita data leakage tra fold durante le trasformazioni OHE
    param_distributions=rf_param_distributions,
    n_iter=30,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    refit=True, # Una volta trovati i miglior parametri per F1, alleniami il modello con tutti i dati di train, non solo quello dei fold
    verbose=1,
    return_train_score=False
)

print(rf_search)

# ------
# Eseguire il tuning della Random Forest

rf_search.fit(X_train, y_train) # Tuning e training completati

# -------
# Mostra iperparametri, F1 medio per fold e pipeline

print("Migliori iperparametri trovati per f1:")
print(rf_search.best_params_)

print(f"\nMiglior F1 medio in CV: {rf_search.best_score_:.4f}")
# Miglior F1 medio in CV: 0.5820


best_rf_pipeline = rf_search.best_estimator_ # Pipeline e Modello allenato(refit=True) con i params di RandomizeSearch

print("\nPipeline selezionata:")
print(best_rf_pipeline)

# --------

# Misurare la stabilità dei risultati
# Vogliamo sapere se il punteggio è rimasto simile nei cinque fold oppure se varia molto in base ai dati usati per la validazione.

cv_results_rf = pd.DataFrame(rf_search.cv_results_)

best_idx = rf_search.best_index_

print("F1 medio:", round(cv_results_rf.loc[best_idx, "mean_test_score"], 4))
print("Deviazione standard F1:", round(cv_results_rf.loc[best_idx, "std_test_score"], 4))

print("\nF1 sui singoli fold:")
for i in range(5):
    score = cv_results_rf.loc[best_idx, f"split{i}_test_score"]
    print(f"Fold {i + 1}: {score:.4f}")

# F1 medio: 0.582
# Deviazione standard F1: 0.0283 # relativamente contenuta, pur non essendo trascurabile. 
# Una deviazione standard F1 abbastanza grande puo' essere un indizio di overfitting

# --------

# Eseguiamo un ulteriore tuning manuale di max_features mantenendo fissi i parametri ottenuti con RandomizedSearchCV:
# Il tuning verra selezionato con cross_validate, in modo da lasciare il test set indipendente, e usarlo solo 
# per valutare le prestazioni del modello finito.

# max_features: 

from sklearn.base import clone # non modifichiamo ripetutamente la pipeline, creiamo invece una nuova pipeline per ogni configurazione, riutilizzando il preprocessor.
# l'import clone(preprocessor), fa' in modo che il preprocessor sia riutilizzabile, facendo 'dimenticare' al preprocessor i precedenti fit.transform del train set. 
# Questo evita data leakage

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
        "F1 medio": scores["test_f1"].mean(), # test_f1 e' la key-word usata da cross_validate
        "F1 std": scores["test_f1"].std(),
        "Precision media": scores["test_precision"].mean(),
        "Recall medio": scores["test_recall"].mean()
    })

results_df = pd.DataFrame(results)
print(results_df.round(4).to_string(index=False))

# None e' risultata il miglior iperparametro value.
# F1 std = 0.0283 = deviazione standard bassa ma da considerare.

# -----
# Facciamo un altro tuning manuale su min_samples_split e min_samples_leaf, con la stessa logica che abbiamo usato per max_features: 

# Valori da confrontare
split_values = [3, 5, 10, 15]
leaf_values = [2, 4, 6, 10]

results = []

for min_split in split_values:
    for min_leaf in leaf_values:

        model = RandomForestClassifier(
            n_estimators=269,
            max_depth=10,
            min_samples_split=min_split, # 6 -> nr favorito per miglior f1
            min_samples_leaf=min_leaf, # 15 -> nr favorito per miglior f1
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

# Consolidiamo il modello con gli iperparametri ricavati con RandomizeSearchCv e con i tuning manuali che abbiamo applicato usando cross-validate.
# Dopo eseguiamo la valutazione finale sul test set 

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
y_proba = rf_final.predict_proba(X_test)

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

#       Accuracy   Precision  Recall   F1        ROC-AUC
#        0.7956    0.6378     0.5321   0.5802    0.8392
    