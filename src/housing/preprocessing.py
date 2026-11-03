from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from housing.config import DROP_COLUMNS, TARGET_COLUMN


def identify_feature_types(X):
    """
    Identify numerical and categorical features.

    MS SubClass is explicitly treated as categorical,
    despite being stored as a numerical column.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature DataFrame.

    Returns
    -------
    tuple
        Numerical feature names and categorical feature names.
    """
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    # Treat MS SubClass as categorical
    if "MS SubClass" in numeric_features:
        numeric_features.remove("MS SubClass")
        categorical_features.append("MS SubClass")

    return numeric_features, categorical_features


def build_preprocessor(X):
    """
    Build the preprocessing pipeline.

    Numerical features:
        - Median imputation

    Categorical features:
        - Constant imputation using 'Missing'
        - One-hot encoding with unknown category handling

    Parameters
    ----------
    X : pd.DataFrame
        Training feature DataFrame used to identify feature types.

    Returns
    -------
    ColumnTransformer
        Complete preprocessing pipeline.
    """
    numeric_features, categorical_features = identify_feature_types(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Missing",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor
