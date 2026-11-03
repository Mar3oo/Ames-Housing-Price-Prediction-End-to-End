import joblib
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from housing.config import (
    MODEL_DIR,
    MODEL_PATH,
    MODEL_NAME,
    MODEL_PARAMS,
    BEST_CV_RMSE,
)
from housing.data import prepare_data
from housing.preprocessing import build_preprocessor
from housing.experiment_tracking import setup_mlflow


def build_model_pipeline(X_train):
    """
    Build the complete preprocessing and modeling pipeline.
    """
    preprocessor = build_preprocessor(X_train)

    model = GradientBoostingRegressor(**MODEL_PARAMS)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_model(X_train, y_train):
    """
    Fit the complete preprocessing and modeling pipeline.
    """
    pipeline = build_model_pipeline(X_train)

    pipeline.fit(X_train, y_train)

    return pipeline


def save_model(model, path=MODEL_PATH):
    """
    Save the fitted pipeline locally using Joblib.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)

    print(f"Model saved successfully: {path}")


def main():
    """
    Execute model training and log the run to MLflow.
    """
    print("=" * 60)
    print(f"Training {MODEL_NAME}")
    print("=" * 60)

    # 1. Configure MLflow
    setup_mlflow()

    # 2. Load and prepare data
    print("\nLoading and preparing data...")

    data = prepare_data()

    X_train = data["X_train_clean"]
    y_train = data["y_train_clean"]

    excluded_count = len(data["X_train"]) - len(X_train)

    print(f"Training observations: {len(X_train)}")
    print(f"Training features: {X_train.shape[1]}")
    print(f"Excluded observations: {excluded_count}")

    # 3. Start an MLflow training run
    with mlflow.start_run(run_name="gradient-boosting-training") as run:
        print(f"\nMLflow Run ID: {run.info.run_id}")

        # Log model configuration
        mlflow.log_params(MODEL_PARAMS)

        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("training_features", X_train.shape[1])
        mlflow.log_param("excluded_observations", excluded_count)

        # Log experiment metadata
        mlflow.set_tags(
            {
                "model_type": "regression",
                "algorithm": "gradient_boosting",
                "dataset": "ames_housing",
                "stage": "training",
            }
        )

        # Record the CV benchmark from experimentation
        mlflow.log_metric(
            "best_cv_rmse",
            BEST_CV_RMSE,
        )

        # 4. Train model
        print("\nBuilding and fitting model pipeline...")

        model = train_model(X_train, y_train)

        print("Model training completed.")

        # 5. Save local model for existing inference workflow
        save_model(model)

        # 6. Log fitted model to MLflow
        print("\nLogging model artifact to MLflow...")

        input_example = X_train.head(3)
        predictions = model.predict(input_example)

        signature = infer_signature(
            input_example,
            predictions,
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            signature=signature,
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

        print("Model artifact logged successfully.")

        print(f"\nMLflow Run ID: {run.info.run_id}")

    print("\nTraining workflow completed successfully.")


if __name__ == "__main__":
    main()
