import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from housing.config import MODEL_PARAMS
from housing.predict import load_model
from housing.train import build_model_pipeline, save_model


def test_build_model_pipeline_uses_configured_gradient_boosting(synthetic_features):
    pipeline = build_model_pipeline(synthetic_features)

    assert isinstance(pipeline, Pipeline)
    model = pipeline.named_steps["model"]
    assert isinstance(model, GradientBoostingRegressor)
    for parameter, expected in MODEL_PARAMS.items():
        assert model.get_params()[parameter] == expected


def test_train_model_fits_pipeline_and_can_save_and_load(
    fitted_model, synthetic_features, tmp_path
):
    predictions = fitted_model.predict(synthetic_features)
    assert predictions.shape == (len(synthetic_features),)
    assert np.isfinite(predictions).all()

    model_path = tmp_path / "fitted_pipeline.joblib"
    save_model(fitted_model, model_path)
    loaded_model = load_model(model_path)

    np.testing.assert_allclose(
        loaded_model.predict(synthetic_features),
        predictions,
    )
