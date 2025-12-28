import mlflow
import mlflow.lightgbm


def setup_mlflow(experiment_name: str = "gaming-ltv"):
    """Setup MLflow experiment"""
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(experiment_name)
    mlflow.lightgbm.autolog(
        log_models=False, log_datasets=False, disable=False, silent=True
    )


def load_latest_model(model_name: str, experiment_name: str = "gaming-ltv"):
    """Load the latest version of a model from MLflow"""
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    latest_run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{latest_run_id}/{model_name}"

    return mlflow.lightgbm.load_model(model_uri)
