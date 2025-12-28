from gaming_lifetimevalue.jobs.train_cohort_classifier import train_cohort_classifier
from gaming_lifetimevalue.jobs.train_cohort_regressor import train_cohort_regressor
from gaming_lifetimevalue.jobs.evaluate_models import evaluate_models
from gaming_lifetimevalue.utils.config_loader import load_config
import mlflow
from gaming_lifetimevalue.utils.mlflow_utils import setup_mlflow
from pathlib import Path
import polars as pl


def main():
    # load parameters from confs/params.yml
    params = load_config()

    # Load training and test data
    train_data = pl.read_parquet(
        Path(params["processed_data_path"]) / "train_data.parquet"
    )
    test_data = pl.read_parquet(
        Path(params["processed_data_path"]) / "test_data.parquet"
    )

    setup_mlflow(experiment_name="gaming-ltv")

    with mlflow.start_run(run_name="two_step_pipeline"):
        # Train cohort classifier
        classifier = train_cohort_classifier(
            train_df=train_data.drop([params["target_column"]]),
            lgbm_params=params["lgbm_classifier_params"],
            cat_cols=params["categorical_columns"],
            target_map=params["target_map"],
        )

        mlflow.lightgbm.log_model(classifier, name="classifier")

        # Train cohort regressors
        cohort_regressors = {}
        cohorts = params["target_map"].keys()

        for cohort_name in cohorts:
            if cohort_name == "No Revenue":
                continue
            cohort_data = train_data.filter(train_data["cohort"] == cohort_name)

            if len(cohort_data) > 0:
                regressor = train_cohort_regressor(
                    train_df=cohort_data,
                    lgbm_params=params["lgbm_regressor_params"],
                    cohort_name=cohort_name,
                    cat_cols=params["categorical_columns"],
                )
                model_name = (
                    f"regressor_{cohort_name.replace(' ', '_').replace('%', 'pct')}"
                )
                mlflow.lightgbm.log_model(regressor, name=model_name)
                cohort_regressors[cohort_name] = regressor
        
        # Evaluate pipeline on test set
        
        eval_metrics = evaluate_models(
            test_df=test_data,
            classifier=classifier,
            cohort_regressors=cohort_regressors,
            cat_cols=params["categorical_columns"],
            target_col=params["target_column"],
            target_map=params["target_map"],
        )
        
        mlflow.log_metric("test_classifier_accuracy", eval_metrics["classifier"]["accuracy"])
        mlflow.log_metric("test_classifier_f1", eval_metrics["classifier"]["f1_weighted"])
        mlflow.log_metric("test_mae", eval_metrics["regressor"]["mae"])
        mlflow.log_metric("test_rmse", eval_metrics["regressor"]["rmse"])
        mlflow.log_metric("test_r2", eval_metrics["regressor"]["r2"])


if __name__ == "__main__":
    main()
