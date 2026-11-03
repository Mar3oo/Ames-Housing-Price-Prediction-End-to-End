import mlflow

# Local MLflow tracking server
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# Main project experiment
EXPERIMENT_NAME = "ames-house-price-prediction"


def setup_mlflow():
    """
    Configure MLflow tracking and select the project experiment.

    Returns
    -------
    Experiment
        The active MLflow experiment.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)

    return experiment
