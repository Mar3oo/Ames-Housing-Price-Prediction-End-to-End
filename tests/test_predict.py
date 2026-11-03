import numpy as np
import pandas as pd
import pytest

from housing.predict import predict_price


def test_predict_price_handles_one_or_multiple_rows_and_returns_finite_values(
    fitted_model, synthetic_features
):
    one_row = synthetic_features.iloc[[0]].copy()
    one_prediction = predict_price(one_row, fitted_model)
    multiple_predictions = predict_price(synthetic_features.iloc[:4].copy(), fitted_model)

    assert len(one_prediction) == 1
    assert one_prediction.index.equals(one_row.index)
    assert len(multiple_predictions) == 4
    assert pd.api.types.is_numeric_dtype(multiple_predictions)
    assert np.isfinite(multiple_predictions).all()


def test_predict_price_reorders_input_columns_to_training_order(
    fitted_model, synthetic_features
):
    sample = synthetic_features.iloc[[0]].copy()
    reversed_sample = sample.loc[:, list(reversed(sample.columns))]

    expected = predict_price(sample, fitted_model)
    actual = predict_price(reversed_sample, fitted_model)

    pd.testing.assert_series_equal(actual, expected)


def test_predict_price_reports_missing_features(fitted_model, synthetic_features):
    incomplete = synthetic_features.iloc[[0]].drop(columns="Lot Area")

    with pytest.raises(ValueError, match="Missing required features"):
        predict_price(incomplete, fitted_model)


def test_predict_price_reports_unexpected_features(fitted_model, synthetic_features):
    extra = synthetic_features.iloc[[0]].assign(UnexpectedFeature=1)

    with pytest.raises(ValueError, match="Unexpected features"):
        predict_price(extra, fitted_model)
