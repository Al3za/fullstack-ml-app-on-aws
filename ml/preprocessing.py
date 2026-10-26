
# PREPROCESSING

# creiamo funzioni di preprocessing reutilizzabili

from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Categorical encoder
def make_categorical_encoder():
    return OneHotEncoder(
    handle_unknown="ignore", # non blocca l'encoder se si presenta una categoria nuova durente l'inferenza
    sparse_output=False, # sparse_output=True utile con molte categorie
    # drop="first" Non usiamo questa metrica perche andremo ad usare ensemble e non un modello lineare, e in questi e preferibile preservare tutte le features categoriali convertite in nr
)

# -----
# -----

from sklearn.compose import ColumnTransformer

# 2. ColumnTransformer
def make_preprocessor(
    numeric_features,
    binary_features,
    categorical_features,
    scale_numeric=False
):
    categorical_encoder = make_categorical_encoder()

    numeric_transformer=(
        StandardScaler() if scale_numeric else "passthrough"
    )

    kwargs = {}

    if scale_numeric:
        kwargs = {
            "remainder": "drop",
            "verbose_feature_names_out": True
        }

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer, # Per distinguere tree model da linear model (Linear ha bisogno della scalatura dei dati mentre i tree model no)
                numeric_features
            ),
            (
                "binary",
                "passthrough",
                binary_features
            ),
            (
                "categorical",
                categorical_encoder,
                categorical_features
            )
        ],
        **kwargs
    )

    return preprocessor