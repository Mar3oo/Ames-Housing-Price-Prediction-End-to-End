import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer

from housing.preprocessing import build_preprocessor, identify_feature_types


def test_identify_feature_types_treats_ms_subclass_as_categorical():
    features = pd.DataFrame(
        {
            "Lot Area": [8000, 9000],
            "Overall Qual": [5, 7],
            "Neighborhood": ["NAmes", "CollgCr"],
            "MS SubClass": [20, 60],
        }
    )

    numeric, categorical = identify_feature_types(features)

    assert numeric == ["Lot Area", "Overall Qual"]
    assert categorical == ["Neighborhood", "MS SubClass"]


def test_build_preprocessor_returns_column_transformer(synthetic_features):
    preprocessor = build_preprocessor(synthetic_features)

    assert isinstance(preprocessor, ColumnTransformer)
    assert {name for name, _, _ in preprocessor.transformers} == {"num", "cat"}


def test_preprocessor_handles_missing_and_unseen_values(synthetic_features):
    preprocessor = build_preprocessor(synthetic_features)
    preprocessor.fit(synthetic_features)

    incoming = pd.DataFrame(
        {
            "Lot Area": [np.nan, 9000],
            "Overall Qual": [5, np.nan],
            "MS SubClass": [20, 30],
            "Neighborhood": [None, "Previously Unseen"],
        }
    )
    transformed = preprocessor.transform(incoming)
    dense = transformed.toarray() if sparse.issparse(transformed) else np.asarray(transformed)

    assert dense.shape[0] == len(incoming)
    assert np.isfinite(dense).all()
