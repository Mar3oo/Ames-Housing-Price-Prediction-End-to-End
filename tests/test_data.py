import numpy as np
import pandas as pd
import pytest

from housing.config import EXCLUDED_TRAIN_INDICES
from housing.data import (
    load_data,
    prepare_data,
    split_data,
    split_features_target,
)


def test_load_data_reads_valid_csv(tmp_path):
    expected = pd.DataFrame(
        {
            "SalePrice": [150000, 210000],
            "Lot Area": [8000, 10000],
        }
    )
    path = tmp_path / "ames.csv"
    expected.to_csv(path, index=False)

    loaded = load_data(path)

    pd.testing.assert_frame_equal(loaded, expected)


def test_load_data_reports_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data(tmp_path / "missing.csv")


def test_split_features_target_removes_target_and_excluded_columns():
    data = pd.DataFrame(
        {
            "SalePrice": [100000, 200000],
            "Lot Area": [7000, 9000],
            "Order": [1, 2],
            "PID": [123, 456],
            "LogSalePrice": [11.5, 12.2],
        }
    )

    features, target = split_features_target(data)

    assert features.columns.tolist() == ["Lot Area"]
    pd.testing.assert_series_equal(target, data["SalePrice"])


def test_train_test_split_is_reproducible():
    features = pd.DataFrame({"feature": np.arange(30)})
    target = pd.Series(np.arange(30) * 10)

    first_split = split_data(features, target)
    second_split = split_data(features, target)

    for first, second in zip(first_split, second_split):
        if isinstance(first, pd.DataFrame):
            pd.testing.assert_frame_equal(first, second)
        else:
            pd.testing.assert_series_equal(first, second)


def test_prepare_data_excludes_training_anomalies_and_preserves_test_data(tmp_path):
    row_count = max(EXCLUDED_TRAIN_INDICES) + 320
    source = pd.DataFrame(
        {
            "SalePrice": np.arange(row_count) * 1000,
            "Lot Area": np.arange(row_count) + 5000,
        }
    )
    path = tmp_path / "synthetic_ames.csv"
    source.to_csv(path, index=False)

    prepared = prepare_data(path)
    features, target = split_features_target(prepared["data"])
    expected_X_train, expected_X_test, expected_y_train, expected_y_test = split_data(features, target)
    selected_train_indices = set(EXCLUDED_TRAIN_INDICES) & set(expected_X_train.index)
    selected_test_indices = set(EXCLUDED_TRAIN_INDICES) & set(expected_X_test.index)

    assert selected_train_indices
    assert prepared["X_train_clean"].index.equals(expected_X_train.drop(index=selected_train_indices).index)
    assert prepared["y_train_clean"].index.equals(expected_y_train.drop(index=selected_train_indices).index)
    pd.testing.assert_frame_equal(prepared["X_test"], expected_X_test)
    pd.testing.assert_series_equal(prepared["y_test"], expected_y_test)
    assert selected_test_indices <= set(prepared["X_test"].index)
