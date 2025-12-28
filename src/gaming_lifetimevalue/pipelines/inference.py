from gaming_lifetimevalue.jobs.preprocess_input_data import preprocess_input_data
from gaming_lifetimevalue.jobs.infer_cohort_classifier import infer_cohort_classifier
from gaming_lifetimevalue.jobs.infer_cohort_regressor import infer_cohort_regressor
from gaming_lifetimevalue.utils.config_loader import load_config
from gaming_lifetimevalue.utils.mlflow_utils import load_latest_model
import polars as pl
from pathlib import Path


def main():
    params = load_config()

    raw_test_data = pl.read_parquet("data/raw/test_samples.parquet")

    preprocessed_data = preprocess_input_data(
        pl_df=raw_test_data, cat_cols=params["categorical_columns"]
    )

    classifier = load_latest_model("classifier")
    
    cohort_regressors = {}
    cohorts = params["target_map"].keys()
    
    for cohort_name in cohorts:
        if cohort_name == "No Revenue":
            cohort_regressors[cohort_name] = None
        else:
            model_name = f"regressor_{cohort_name.replace(' ', '_').replace('%', 'pct')}"
            try:
                cohort_regressors[cohort_name] = load_latest_model(model_name)
            except Exception as e:
                print(f"Warning: Could not load {model_name}: {e}")

    data_with_cohort = infer_cohort_classifier(
        classifier, preprocessed_data, params["target_map"]
    )

    all_predictions = []
    
    for cohort_name, regressor in cohort_regressors.items():
        cohort_data = data_with_cohort.filter(
            pl.col("predicted_cohort") == cohort_name
        )
        
        if len(cohort_data) == 0:
            continue
        
        cohort_with_pred = infer_cohort_regressor(
            regressor, cohort_data, params["categorical_columns"]
        )
        all_predictions.append(cohort_with_pred)
    
    final_predictions = pl.concat(all_predictions)

    output_path = Path(params.get("predictions_path", "data/predictions")) / "test_predictions.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_predictions.write_parquet(output_path)


if __name__ == "__main__":
    main()
