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

for columns in categorical_features:
    print(f"\n--- {columns} ---")

    print("Number of unique values:")
    print(df_analysis[columns].nunique())

    print("Values:")
    print(df_analysis[columns].value_counts())


print("=== CHECK CATEGORICAL WHITESPACE ===")
# importante verificare se ci sono categorie uguali ma scritte diversamente, ad esempio con aggiunte 
# di spazzi ("yes" e " Yes") oppure con Upper/LowerCase ("yes" e "Yes"). questo potrebbe creare colonne aggiuntive
# quando usiamo OneHotEncoder, e confondere il modello. le function df_analysis[column].unique() e .valueCounts
#  gia' ci mostra se ci sono problemi del genere, ma applica queste ulteriori misure di sicurezza 
for columns in categorical_features:

    values = df_analysis[columns].astype("string")

    # Conta i valori che cambiano dopo la rimozione degli spazi (" yes " != "yes" -> true)
    whitespace_count = (values != values.str.strip()).sum()

    # 2. Normalizzazione temporanea. Porta tutte le values in Lowercase
    # normalized = values.str.strip().str.lower()

    # check lower case as well

    print(f"\n--- {columns} ---")
    print(f"Values with leading/trailing spaces: {whitespace_count}")
    
    
    


   
