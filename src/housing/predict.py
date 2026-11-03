import joblib
import pandas as pd

from housing.config import MODEL_PATH


def load_model(path=MODEL_PATH):
    """
    Load the trained model pipeline.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run the training module first."
        )

    return joblib.load(path)


def validate_input(input_data, model):
    """
    Validate prediction input against the model's expected features.

    Parameters
    ----------
    input_data : pd.DataFrame
        Raw house features.
    model : Pipeline
        Fitted model pipeline.

    Returns
    -------
    pd.DataFrame
        Validated input DataFrame.
    """
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if input_data.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    expected_features = model.feature_names_in_

    missing_features = set(expected_features) - set(input_data.columns)
    unexpected_features = set(input_data.columns) - set(expected_features)

    if missing_features:
        raise ValueError(f"Missing required features: {sorted(missing_features)}")

    if unexpected_features:
        raise ValueError(f"Unexpected features: {sorted(unexpected_features)}")

    # Ensure the same column order used during training
    input_data = input_data.loc[:, expected_features]

    return input_data


def predict_price(input_data, model=None):
    """
    Predict sale prices for one or more houses.

    Parameters
    ----------
    input_data : pd.DataFrame
        Raw house features, excluding SalePrice and identifiers.
    model : Pipeline, optional
        Fitted model pipeline. Loaded automatically if omitted.

    Returns
    -------
    pd.Series
        Predicted sale prices.
    """
    if model is None:
        model = load_model()

    input_data = validate_input(input_data, model)

    predictions = model.predict(input_data)

    return pd.Series(
        predictions,
        index=input_data.index,
        name="Predicted Sale Price",
    )


def main():
    """
    Example prediction using a row from the original dataset.

    This is only a demonstration of the inference workflow.
    """
    from housing.data import load_data, split_features_target

    print("=" * 60)
    print("House Price Prediction")
    print("=" * 60)

    # Load model
    model = load_model()

    # Load an example house from the dataset
    df = load_data()

    X, y = split_features_target(df)

    sample = X.iloc[[0]]

    # Predict
    prediction = predict_price(sample, model)

    print("\nPredicted Sale Price:")
    print(f"${prediction.iloc[0]:,.2f}")

    print("\nActual Sale Price:")
    print(f"${y.iloc[0]:,.2f}")


if __name__ == "__main__":
    main()
