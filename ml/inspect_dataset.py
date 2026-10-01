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

#analisi sulle colonne TotalCharges e churn
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
# 2. Il target è sbilanciato, ma non in modo estremo, qundi il parametro accuracy da solo e' forviante, quindi andremo ad usare anche metriche come:

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


## alcune features binarie:gender
# Partner
# Dependents
# PhoneService
# PaperlessBilling

## alcune features categoriche con più categorie (one hot encoding):
# InternetService,
# Contract
# PaymentMethod 


# check teanure and other columns for the 11 clients with empty("") TotalCharges values:

empty_total_charges = df['TotalCharges'].str.strip() == ""

print("/n rows with empty TotalCharges:")
print(
    df.loc[
        empty_total_charges,
        ["customerID","tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ]
) 


# make sure that tenure = 0 and TotalCharges == "" are related:
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



## transform TotalCharges empty data "" to 0 to perform .describe() on the column:

# create a copy of the dataset. (lascia sempre i dati originali invariati, e fai le trasformazioni su una copia):
df_analysis = df.copy()

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


# Analisi degli outlier nelle variabili numeriche

print("=== TASK 2: OUTLIER ANALYSIS ===")

numeric_features = df_analysis.select_dtypes(include="number").columns.drop("SeniorCitizen") # SeniorCitizen perche non ci sono outliers in una colonna composta da 0-1
numeric_features # check SeniorCitizen is dropped

# calcolo Iqr per verificare se ci sono eventuali outliers:for column in numeric_features: # i dati centrali
for column in numeric_features: # i dati centrali
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

# without ['customerID', SeniorCitizen, 'TotalCharges']
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

# TASK 4 — Target Analysis

   
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


# un passo ulteriore:facciamo una piccola verifica quantitativa sulle associazioni categoriche: quanto sono statisticamente associate a Churn?

# Useremo il Chi-square test. Non serve ancora conoscere tutta la statistica dietro al test: per ora ci 
# interessa capire perché lo usiamo.


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

# descrizione output in Task5.md

# -------------

print("=== TASK 5.4: NUMERIC FEATURES vs CHURN ===")

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
# confronta le distribuzioni delle variabili numeriche tra i due gruppi di Churn.
# desc in task5+.md