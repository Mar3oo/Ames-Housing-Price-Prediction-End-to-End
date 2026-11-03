import mlflow

from housing.experiment_tracking import setup_mlflow


def main():

    setup_mlflow()

    with mlflow.start_run(run_name="first-mlflow-run"):
        # Log model parameters
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_param("n_estimators", 363)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("learning_rate", 0.06156)

        # Log evaluation metrics (illustrative historical results)
        mlflow.log_metric("test_mae", 13969.872)
        mlflow.log_metric("test_rmse", 28864.293)
        mlflow.log_metric("test_r2", 0.896)

        # Log a descriptive tag
        mlflow.set_tag("stage", "initial_tracking_test")

        print("First MLflow run logged successfully!")


if __name__ == "__main__":
    main()
