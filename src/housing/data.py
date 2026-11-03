import pandas as pd

from sklearn.model_selection import train_test_split

from housing.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    DROP_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
    EXCLUDED_TRAIN_INDICES,
)


def load_data(path=RAW_DATA_PATH):
    """
    Load the raw Ames Housing dataset.

    Parameters
    ----------
    path : Path
        Path to the raw CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    return df


def split_features_target(df):
    """
    Separate input features from the target variable.

    Identifier columns and previously engineered target
    columns are excluded from the predictors.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset.

    Returns
    -------
    tuple
        X: Feature DataFrame.
        y: Target Series.
    """
    columns_to_drop = [
        TARGET_COLUMN,
        *DROP_COLUMNS,
    ]

    existing_drop_columns = [col for col in columns_to_drop if col in df.columns]

    X = df.drop(columns=existing_drop_columns)
    y = df[TARGET_COLUMN].copy()

    return X, y


def split_data(X, y):
    """
    Split the dataset into training and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Input features.

    y : pd.Series
        Target variable.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test.
    """
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def exclude_anomalous_observations(X_train, y_train):
    """
    Exclude selected anomalous observations from training data.

    The original test set is not affected.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training target.

    Returns
    -------
    tuple
        Cleaned training features and target.
    """
    indices_to_exclude = [idx for idx in EXCLUDED_TRAIN_INDICES if idx in X_train.index]

    X_train_clean = X_train.drop(index=indices_to_exclude)

    y_train_clean = y_train.drop(index=indices_to_exclude)

    return X_train_clean, y_train_clean


def prepare_data(path=RAW_DATA_PATH):
    """
    Execute the complete data preparation workflow.

    Returns
    -------
    dict
        Original data, training/test splits, and cleaned
        training data.
    """
    df = load_data(path)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_clean, y_train_clean = exclude_anomalous_observations(
        X_train,
        y_train,
    )

    return {
        "data": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_clean": X_train_clean,
        "y_train_clean": y_train_clean,
    }
