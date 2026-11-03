import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from housing.evaluate import (
    calculate_metrics,
    evaluate_model,
    get_largest_prediction_errors,
)


def test_calculate_metrics_returns_mae_rmse_and_r2():
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([110.0, 190.0, 330.0])

    metrics = calculate_metrics(actual, predicted)

    assert metrics["MAE"] == pytest.approx(50 / 3)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(1100 / 3))
    assert metrics["R2"] == pytest.approx(0.945)


def test_largest_prediction_errors_include_signed_residual_and_absolute_error():
    errors = get_largest_prediction_errors(
        y_true=[100, 200, 300],
        y_pred=[110, 180, 250],
        top_n=2,
    )

    assert errors["Actual Price"].tolist() == [300, 200]
    assert errors["Predicted Price"].tolist() == [250, 180]
    assert errors["Residual"].tolist() == [50, 20]
    assert errors["Absolute Error"].tolist() == [50, 20]


def test_evaluate_model_returns_predictions_and_expected_metrics():
    features = pd.DataFrame({"feature": [0, 1]})
    model = DummyRegressor(strategy="constant", constant=150).fit(features, [150, 150])

    predictions, metrics = evaluate_model(model, features, [100, 200])

    np.testing.assert_array_equal(predictions, [150, 150])
    assert metrics == {"MAE": 50.0, "RMSE": 50.0, "R2": 0.0}
