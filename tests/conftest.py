import numpy as np
import pandas as pd
import pytest

from housing.train import train_model


@pytest.fixture(scope="session")
def synthetic_features():
    return pd.DataFrame(
        {
            "Lot Area": [8500, 9600, np.nan, 11200, 7400, 10100, 8800, 13200, 9100, 7800, 10500, 11700],
            "Overall Qual": [5, 6, 7, 8, 4, 6, 5, 9, 7, 5, 8, 6],
            "MS SubClass": [20, 60, 20, 120, 30, 60, 20, 120, 60, 30, 20, 60],
            "Neighborhood": ["NAmes", "CollgCr", "NAmes", None, "Edwards", "CollgCr", "Edwards", "NAmes", "CollgCr", "NAmes", "Edwards", "CollgCr"],
        }
    )


@pytest.fixture(scope="session")
def synthetic_target(synthetic_features):
    return pd.Series(
        [135000, 178000, 205000, 245000, 118000, 190000, 152000, 275000, 215000, 142000, 230000, 198000],
        index=synthetic_features.index,
        name="SalePrice",
    )


@pytest.fixture(scope="session")
def fitted_model(synthetic_features, synthetic_target):
    return train_model(synthetic_features, synthetic_target)
