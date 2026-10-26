# Analisi  dati e trasformazioni iniziali

import pandas as pd


DATA_PATH = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("Shape:")
print(df.shape) # the shape of the data (7043, 21) Rows and col

print("\nColumns:")
print(df.columns.tolist()) # shows a list of the columns

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())



print("\ncharges:")
print(df["TotalCharges"].dtype)

print("\ncharges should be numeric but is str:")
print(df["TotalCharges"])

# Analisi sulle colonne TotalCharges e churn
# mostra 20 valori univoci dalla colonna TotalCharges. 
print("\nUnique values in TotalCharges:")
print(df["TotalCharges"].unique()[:20])


# 11 Empty strings in TotalCharges (not null but empty "")
print("\nEmpty strings in TotalCharges:")
print((df["TotalCharges"].str.strip() == "").sum()) # controlla quali valori, 
# dopo aver tolto gli spazi, risultano stringhe vuote.

# how many no and yes in churn column?
print("\nTarget distribution:")
print(df["Churn"].value_counts()) 

# make a percentual of this value count (73.463013 % no and 26.536987 yes )
print("\nTarget distribution (%):")
print(df["Churn"].value_counts(normalize=True) * 100) # con normalize=True si 
# ottiene invece la frazione sul totale:


## tre decisioni di preprocessing:
# Colonna   	Problema	                Cosa faremo
# customerID	identificativo	            probabilmente escluderla
# TotalCharges	stringa + 11 valori vuoti	convertire/imputare
# Churn	        target categorico	        trasformare in target binario

# Le altre colonne le analizzeremo prima di decidere cosa fare.


# riguardo il target:
# 2. Il target è sbilanciato, ma non in modo estremo (74%-26%). Il parametro accuracy da solo e' forviante, quindi andremo ad usare anche metriche come:

# precision
# recall
# F1-score
# confusion matrix
# ROC-AUC


## controllare le categorie presenti nelle colonne categoriche.
print("\nCategorical columns and unique values:") # 

# ricaviamo tutte le colonne con dati categorici:
categoricals_columns = df.select_dtypes(include=["object"]).columns # shows all <StringArray> columns
print(categoricals_columns)
print("\nnr categorical columns:")
print(len(categoricals_columns))

for column in categoricals_columns:
    print(f"\n {column}")
    print(df[column].unique())
    print('check missing string values')
    print((df[column].str.strip() == "").sum())


## alcune features binarie:
# gender
# Partner
# Dependents
# PhoneService
# PaperlessBilling

## alcune features categoriche con più categorie (one hot encoding):
# InternetService,
# Contract
# PaymentMethod 


# Controllo dati degli 11 clienti con TotalCharges = ""

empty_total_charges = df['TotalCharges'].str.strip() == ""

print("/n rows with empty TotalCharges:")
print(
    df.loc[
        empty_total_charges,
        ["customerID","tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ]
) # Trovata relazione con tenure 


# Assicuriamoci che tenure = 0 e TotalCharges == "" sono correlati  
print("\nCustomers with tenure = 0:")
print(
    df.loc[
        df["tenure"] == 0,
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ]
)

# tenure = 0 and Totalcharges = "". Una possibile interpretazione è che siano clienti appena entrati nel
# servizio: hanno un piano con un costo mensile, ma non hanno ancora accumulato addebiti complessivi. Soluzioni di feature engineering:

# 1) Imputare con la mediana ❌ = non ha senso riempire i campi mancanti di Totalcharges la media degli altri clienti perche' cio 
# significherebbe inventare un ammontare di spesa che probabilmente non rappresenta quel cliente.

# 2)Trasformare il vuoto in 0 ✅ = Questa è un'ipotesi molto più coerente perche effettivamente il Totalcharges 
# cliente di questi nuovi clienti e' = 0

# --------------

## Trasforma i dati mancanti di TotalCharges = "" con il valore 0 per poter fare  .describe() sulle colonne:

# crea una copia del dataset. (lascia sempre i dati originali invariati, e fai le trasformazioni su una copia):
df_analysis = df.copy()

# Trasforma i dati mancanti di TotalCharges
df_analysis["TotalCharges"] = pd.to_numeric( # Convert argument to a numeric type.
    df_analysis["TotalCharges"],
    errors="coerce" # sets the "" values to NAN
)

print(df_analysis["TotalCharges"].isna().sum()) # 11

# check is nan values:
nan_values = df_analysis["TotalCharges"].isna()

print(
    df_analysis.loc[
        nan_values,
        ["customerID",  "tenure",  "MonthlyCharges", "TotalCharges", "Churn"]
    ]
)

## fill nan value with 0
df_analysis["TotalCharges"] = df_analysis["TotalCharges"].fillna(0)

zero_values = df_analysis["TotalCharges"] == 0

print(" ")

print(
    df_analysis.loc[
        zero_values,
        ["customerID",  "tenure",  "MonthlyCharges", "TotalCharges", "Churn"]
    ]
)

print(" ")
# NUMERIC FEATURES DATA UNDERSTANDING
print("NUMERIC FEATURES DATA UNDERSTANDING")

print("\n--- tenure ---")
print(df_analysis["tenure"].describe())

print("\n--- MonthlyCharges ---")
print(df_analysis["MonthlyCharges"].describe())

print(
    df_analysis.loc[
        df_analysis["tenure"] == 0,
        ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ]
)

print("\n--- TotalCharges ---")
print(df_analysis["TotalCharges"].describe())

# SeniorCitizen

print("\n=== SeniorCitizen: binary categorical ===") # 0 and 1 values to describe this column, so .describe() is no needed

print("\nPercentages:")
print(df["SeniorCitizen"].value_counts(normalize=True) * 100) # percentage of the SeniorCitizen

# ----

# Analisi degli outlier nelle variabili numeriche

print("=== TASK 2: OUTLIER ANALYSIS ===")

numeric_features = df_analysis.select_dtypes(include="number").columns.drop("SeniorCitizen") # droppiamo SeniorCitizen perche non ci sono outliers in una colonna composta da 0-1
numeric_features # check SeniorCitizen is dropped

# Calcolo Iqr per verificare se ci sono eventuali outliers o valori estremi: 

for column in numeric_features:
    Q1 = df_analysis[column].quantile(0.25) # df_analysis[column] ritorna righa index e value della colonna, mentre .quantile(0.25) filtra solo il qualtile inferiore (0.25) 
    Q3 = df_analysis[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5*IQR
    upper_bound  = Q3 + 1.5*IQR

    outliers = df_analysis[
        (df_analysis[column] < lower_bound) |
        (df_analysis[column] > upper_bound)
    ]

    # no outliers found. Nessun valore di nessuna delle colonne del lop sfora il calcolo iqr
    
    print(f"\n--- {column} ---")
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    # print(f"outliers: {(outliers)}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Percentage: {len(outliers) / len(df_analysis) * 100:.2f}%")

# ---------------------------------

# Task 3 -> Analisi delle variabili categoriche

# Ps categorical_features dovrebbe contenere le feature categoriche che utilizzeremo 
# per predire Churn, per questo churn non la inseriamo pur essendo categorica.
# Churn verrà analizzato separatamente per sapere: 
# distribuzione Yes/No
# percentuali
# relazione con le altre variabili

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

print("=== CHECK CATEGORICAL WHITESPACE ===")
for columns in categorical_features:
    print(f"\n--- {columns} ---")

    print("Number of unique values:")
    print(df_analysis[columns].nunique())

    print("Values:")
    print(df_analysis[columns].value_counts())

    # importante verificare se ci sono categorie uguali ma scritte diversamente, ad esempio con aggiunte 
    # di spazzi ("yes" e " Yes") oppure con Upper/LowerCase ("yes" e "Yes"). questo potrebbe creare colonne aggiuntive
    # quando usiamo OneHotEncoder, e confondere il modello. le function df_analysis[column].unique() e .valueCounts
    #  gia' ci mostra se ci sono problemi del genere, ma applica queste ulteriori misure di sicurezza
    values = df_analysis[columns].astype("string")
    whitespace_count = (values != values.str.strip()).sum()

    print(f"Values with leading/trailing spaces: {whitespace_count}")
     
# -------------------

# TASK 4 — Analisi del Target 
   
print("=== TASK 4: TARGET ANALYSIS ===")

target = "Churn"

print("\n--- Target values ---")
print(df_analysis[target].value_counts())

print("\n--- Target percentages ---")
print(df_analysis[target].value_counts(normalize=True) * 100)

print("\n--- Number of unique values ---")
print(df_analysis[target].nunique()) # il nr delle unique values

print("\n--- Missing values ---")
print(df_analysis[target].isna().sum())

print("\n--- Empty string values ---")
print((df_analysis[target].str.strip() == "").sum())

# No     5174 ->  73.46%
# Yes    1869 ->  26.54%
# no Missing values or Empty string values

# moderatamente sbilanciato verso No

# ------------

# TASK 5 — Relazione tra feature e Churn
# Ora che abbiamo analizzato le singole variabili, vogliamo capire quali caratteristiche 
# sembrano essere associate al churn

# Primo sotto-step: feature categoriche vs Churn
# Partiamo dalle categoriche, senza fare ancora grafici.

print("=== TASK 5.1: CATEGORICAL FEATURES vs CHURN ===")

for column in categorical_features:

    print(f"\n--- {column} ---")

    churn_rate = pd.crosstab(
        df_analysis[column],
        df_analysis["Churn"],
        normalize="index"
    ) * 100

    print(churn_rate)

# Cosa emerge

# Ci sono alcune differenze abbastanza marcate nel churn rate tra le categorie:

# gender → differenza molto piccola: ~26–27%
# Partner → No: 33.0% vs Yes: 19.7%
# Dependents → No: 31.3% vs Yes: 15.5%
# InternetService → DSL: 19.0%, Fiber optic: 41.9%, No: 7.4%
# OnlineSecurity → No: 41.8%, Yes: 14.6%
# TechSupport → No: 41.6%, Yes: 15.2%
# Contract → Month-to-month: 42.7%, One year: 11.3%, Two year: 2.8%
# PaperlessBilling → Yes: 33.6% vs No: 16.3%
# PaymentMethod → Electronic check: 45.3%, mentre gli altri metodi sono circa 15–19%

# Al contrario, PhoneService, MultipleLines, StreamingTV e StreamingMovies mostrano differenze più contenute.

# Una cosa importante: queste sono associazioni descrittive, non causalità. Per esempio, possiamo dire che nel
# dataset il churn è più frequente tra i clienti con contratto Month-to-month; non possiamo ancora dire che 
# il contratto mensile causi il churn.



# Facciamo una piccola verifica quantitativa sulle associazioni categoriche: quanto sono statisticamente associate a Churn?
# Useremo il Chi-square test. 

# -----------------

# TASK 5.2 — Test Chi-quadrato

from scipy.stats import chi2_contingency

print("=== TASK 5: CHI-SQUARE TEST ===")

for column in categorical_features:

    contingency_table = pd.crosstab(
        df_analysis[column],
        df_analysis["Churn"]
    )

    chi2, p_value, dof, expected = chi2_contingency(
        contingency_table
    )

    print(f"\n--- {column} ---")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p_value:.6g}")

   # descrizione output su task5.md
    # python inspect_dataset.py


# ------------

# CHI-SQUARE TEST rispondeva a:
# “C’è evidenza di un’associazione tra questa feature e Churn?”

# Il Cramér’s V rispondere a:
# “Quanto è forte questa associazione?”

from scipy.stats import chi2_contingency
import numpy as np

print("=== TASK 5.3: CRAMÉR'S V ===")

for column in categorical_features:

    contingency_table = pd.crosstab(
        df_analysis[column],
        df_analysis["Churn"]
    )

    chi2, p_value, dof, expected = chi2_contingency(
        contingency_table
    )

    n = contingency_table.sum().sum()

    phi2 = chi2 / n

    rows, cols = contingency_table.shape

    cramer_v = np.sqrt(
        phi2 / min(rows - 1, cols - 1)
    )

    print(f"\n--- {column} ---")
    print(f"Cramér's V: {cramer_v:.4f}")

# Alcune var categoriche presentano una possibila' associativa con il target non indifferente.  (Descrizione piu' accurata dell' output in Task5.md)
# -------------

print("=== TASK 5.4: NUMERIC FEATURES vs CHURN ===")

# Confronto di media e mediana di tenure, MonthlyCharges e TotalCharges tra i gruppi Churn = No e Churn = Yes.


numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for column in numeric_features:
    print(f"\n--- {column} ---")
    print(
        df_analysis.groupby("Churn")[column].describe()
    )

# Differenze significative rilevate tra le distribuzioni dei dati delle features verso il taget
# desc in task5+.md

# -----
# TASK 5.5 — Test statistico sulle feature numeriche
# Verifichiamo se le differenze osservate sono statisticamente significative

from scipy.stats import mannwhitneyu

print("=== TASK 5.5: MANN-WHITNEY U TEST ===")

for column in numeric_features:

    group_no = df_analysis.loc[
        df_analysis["Churn"] == "No",
        column
    ]

    group_yes = df_analysis.loc[
        df_analysis["Churn"] == "Yes",
        column
    ]

    statistic, p_value = mannwhitneyu(
        group_no,
        group_yes,
        alternative="two-sided"
    )

    print(f"\n--- {column} ---")
    print(f"Mann-Whitney U statistic: {statistic:.2f}")
    print(f"p-value: {p_value:.6g}")

# Rilevata una evidenza statistica di una differenza tra le distribuzioni dei due gruppi Yes e no. 

# --------

# TASK 5.6 — Visualizzazione delle feature numeriche rispetto a Churn

import matplotlib.pyplot as plt

for column in numeric_features:
    plt.figure(figsize=(8,5)) 

    df_analysis.boxplot(
        column=column,
        by="Churn"
    )

    plt.title(f"{column} vs Churn")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel(column)

    plt.show()


# Le distribuzioni delle tre feature numeriche mostrano differenze visibili 
# tra i gruppi Churn = No e Churn = Yes, coerenti con le differenze già 
# osservate nelle statistiche descrittive e nel Mann–Whitney U test

# ---------

# TASK 5.7 — Correlazione tra le tra le variabili indipendenti. 
# 
# Se prima abbiamo analizzato la correlazioni tra Features e Target ora andremo a vedere se le features indicano una correlazione
# tra le features stesse. Andremo ad usare i calcoli statistici Pearson e spearman per le features numeriche,
# e Chi-square test e cramers per quelle categoriche. 

# Questi dati ci serviranno per fare feature selection in nel con il modello di regressione logistica(logistic_regression.py), 
# e anche per calcolare la Permutation_importance nel nostro modello XGBoost (XGBoost.py)

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
# ----

# Associazioni tra feature categoriche

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


# ------------

# TASK 5.9 — Controllo delle feature ridondanti / logicamente dipendenti

# Qui non facciamo ancora preprocessing. Vogliamo capire se nel dataset esistono
# feature che rappresentano informazioni quasi duplicate o fortemente condizionate 
# da un'altra feature.

print("=== TASK 5.9: LOGICAL DEPENDENCIES ===")

internet_features = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

for column in internet_features:

    print(f"\n--- {column} when InternetService = No ---")

    print(
        df_analysis.loc[
            df_analysis["InternetService"] == "No", # id nr righe filtrate dove InternetService = No
            column # colonne
        ].value_counts()
    )

# La features InternetService con value "No" e' correlata con queste altre features elencate sopra. Dove "No" in una, equivale "No" nelle altre. 

print("=== CHECK: PhoneService -> MultipleLines ===")

print(
    df_analysis.loc[
        df_analysis["PhoneService"] == "No",
        "MultipleLines"
    ].value_counts()
)
# "No phone service" è quindi una categoria semanticamente coerente, e non indica un valore mancante.

# ----------

# TASK 5.10 — Controllo finale di coerenza del dataset 
# Un ultimo sguardo ai dati per assicurarci che non siano rimaste anomalie
#  evidenti nelle colonne categoriche e numeriche

print("=== TASK 5.10: FINAL DATA CONSISTENCY CHECK ===")

print("\n--- Shape ---")
print(df_analysis.shape)

print("\n--- Missing values ---")
print(df_analysis.isna.sum())


print("\n--- Numeric values below zero ---")
print(
    df_analysis[numeric_features]
    .lt(0) # lt stands for les than
    .sum()
)


print("\n--- Empty categorical values ---")
for column in categorical_features:
    print(f'empty categorical value find in {column}:')
    empty_count = (df_analysis[column].str.strip() == "").sum()

    print(f"{column}: {empty_count} ")


print("\n--- Target values ---")
print(df_analysis['Churn'].value_counts())