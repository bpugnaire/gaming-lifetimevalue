from gaming_lifetimevalue.jobs.train_cohort_classifier import train_cohort_classifier
from gaming_lifetimevalue.jobs.train_cohort_regressor import train_cohort_regressor
from gaming_lifetimevalue.jobs.preprocess_input_data import preprocess_input_data
from gaming_lifetimevalue.utils.config_loader import load_config
import mlflow
from gaming_lifetimevalue.utils.mlflow_utils import setup_mlflow


def main():
    # load parameters from confs/params.yml
    params = load_config()

    # Preprocess training data
    processed_data = preprocess_input_data(
        data_path=params["data_path"], cat_cols=params["categorical_columns"]
    )

    setup_mlflow(experiment_name="gaming-ltv")

    with mlflow.start_run(run_name="two_step_pipeline"):
        # Train cohort classifier
        classifier = train_cohort_classifier(
            train_df=processed_data.drop([params["target_column"]]),
            lgbm_params=params["lgbm_classifier_params"],
            cat_cols=params["categorical_columns"],
            target_map=params["target_map"],
        )

        mlflow.lightgbm.log_model(classifier, artifact_path="classifier")

        # Train cohort regressors

        cohorts = params["target_map"].keys()

        for cohort_name in cohorts:
            if cohort_name == "No Revenue":
                continue
            cohort_data = processed_data.filter(processed_data["cohort"] == cohort_name)

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
                mlflow.lightgbm.log_model(regressor, artifact_path=model_name)


if __name__ == "__main__":
    main()
