import tempfile
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from housing.config import MODEL_PATH, BEST_CV_RMSE
from housing.data import prepare_data
from housing.experiment_tracking import setup_mlflow


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Run the training module first."
        )

    return joblib.load(path)


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def create_actual_vs_predicted_plot(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        x=y_true,
        y=y_pred,
        alpha=0.6,
        ax=ax,
    )

    min_value = min(np.min(y_true), np.min(y_pred))
    max_value = max(np.max(y_true), np.max(y_pred))

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        label="Perfect Predictions",
    )

    ax.set_title("Actual vs Predicted Sale Prices")
    ax.set_xlabel("Actual Sale Price")
    ax.set_ylabel("Predicted Sale Price")
    ax.legend()

    fig.tight_layout()
    return fig


def create_residual_plots(y_true, y_pred):
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.scatterplot(
        x=y_pred,
        y=residuals,
        alpha=0.6,
        ax=axes[0],
    )

    axes[0].axhline(y=0, color="red", linestyle="--")
    axes[0].set_title("Residuals vs Predicted Prices")
    axes[0].set_xlabel("Predicted Sale Price")
    axes[0].set_ylabel("Residuals")

    sns.histplot(
        residuals,
        kde=True,
        ax=axes[1],
    )

    axes[1].axvline(x=0, color="red", linestyle="--")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")

    fig.tight_layout()
    return fig


def get_largest_prediction_errors(y_true, y_pred, top_n=15):
    results = pd.DataFrame(
        {
            "Test Index": y_true.index,
            "Actual Price": np.asarray(y_true),
            "Predicted Price": np.asarray(y_pred),
        }
    )

    results["Residual"] = results["Actual Price"] - results["Predicted Price"]

    results["Absolute Error"] = results["Residual"].abs()

    return results.sort_values(
        by="Absolute Error",
        ascending=False,
    ).head(top_n)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    return y_pred, metrics


def main():
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    setup_mlflow()

    print("\nLoading test data...")

    data = prepare_data()
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Test observations: {len(X_test)}")

    print("\nLoading trained model...")
    model = load_model()

    with mlflow.start_run(run_name="gradient-boosting-evaluation") as run:
        print(f"\nMLflow Run ID: {run.info.run_id}")

        print("\nEvaluating model...")
        y_pred, metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        # Log evaluation metrics
        mlflow.log_metrics(
            {
                "test_mae": metrics["MAE"],
                "test_rmse": metrics["RMSE"],
                "test_r2": metrics["R2"],
                "best_cv_rmse": BEST_CV_RMSE,
            }
        )

        mlflow.log_param("test_samples", len(X_test))

        mlflow.set_tags(
            {
                "model_type": "regression",
                "algorithm": "gradient_boosting",
                "dataset": "ames_housing",
                "stage": "evaluation",
            }
        )

        print("\nTest Performance:")

        metrics_df = pd.DataFrame(
            list(metrics.items()),
            columns=["Metric", "Test Score"],
        )

        print(metrics_df.to_string(index=False))

        print(f"\nBest CV RMSE: {BEST_CV_RMSE:,.2f}")
        print(f"Test RMSE:    {metrics['RMSE']:,.2f}")

        # Create and log diagnostic plots
        print("\nCreating diagnostic plots...")

        actual_predicted_fig = create_actual_vs_predicted_plot(
            y_test,
            y_pred,
        )

        residual_fig = create_residual_plots(
            y_test,
            y_pred,
        )

        mlflow.log_figure(
            actual_predicted_fig,
            "plots/actual_vs_predicted.png",
        )

        mlflow.log_figure(
            residual_fig,
            "plots/residual_diagnostics.png",
        )

        # Display plots locally
        plt.show()

        plt.close(actual_predicted_fig)
        plt.close(residual_fig)

        # Generate and log largest prediction errors
        print("\nLargest Prediction Errors:")

        error_df = get_largest_prediction_errors(
            y_test,
            y_pred,
        )

        print(error_df.to_string(index=False))

        with tempfile.TemporaryDirectory() as temp_dir:
            error_path = Path(temp_dir) / "largest_prediction_errors.csv"

            error_df.to_csv(error_path, index=False)

            mlflow.log_artifact(
                str(error_path),
                artifact_path="analysis",
            )

        print("\nEvaluation artifacts logged successfully.")
        print(f"Evaluation Run ID: {run.info.run_id}")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
