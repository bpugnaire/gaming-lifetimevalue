import polars as pl
from gaming_lifetimevalue.jobs.infer_cohort_classifier import infer_cohort_classifier
from gaming_lifetimevalue.jobs.infer_cohort_regressor import infer_cohort_regressor
from gaming_lifetimevalue.evaluation.metrics import evaluate_classifier, evaluate_regressor


def evaluate_pipeline(
    test_df: pl.DataFrame,
    classifier,
    cohort_regressors: dict,
    cat_cols: list,
    target_col: str,
    target_map: dict,
):
    y_true_cohort = (
        test_df.with_columns(pl.col("cohort").replace(target_map).cast(pl.Int64))
        .select("cohort")
        .to_pandas()["cohort"]
    )
    
    test_with_pred = infer_cohort_classifier(classifier, test_df, target_map)
    
    y_pred_cohort = (
        test_with_pred.with_columns(
            pl.col("predicted_cohort").replace(target_map).cast(pl.Int64)
        )
        .select("predicted_cohort")
        .to_pandas()["predicted_cohort"]
    )
    
    classifier_metrics = evaluate_classifier(
        y_true_cohort, y_pred_cohort, target_names=list(target_map.keys())
    )
    
    all_predictions = []
    
    for cohort_name, regressor in cohort_regressors.items():
        cohort_data = test_with_pred.filter(
            pl.col("predicted_cohort") == cohort_name
        )
        
        if len(cohort_data) == 0:
            continue
        
        cohort_with_pred = infer_cohort_regressor(regressor, cohort_data, cat_cols)
        all_predictions.append(cohort_with_pred)
    
    final_predictions = pl.concat(all_predictions)
    
    y_true = final_predictions.select(target_col).to_pandas()[target_col]
    y_pred = final_predictions.select("predicted_d120_rev").to_pandas()["predicted_d120_rev"]
    
    regressor_metrics = evaluate_regressor(y_true, y_pred)
    
    print("Classifier Evaluation:")
    print(f"Accuracy: {classifier_metrics['accuracy']:.4f}")
    print(f"F1 Weighted: {classifier_metrics['f1_weighted']:.4f}")
    
    print(f"\nRegressor Evaluation:")
    print(f"MAE: {regressor_metrics['mae']:.4f}")
    print(f"RMSE: {regressor_metrics['rmse']:.4f}")
    print(f"R²: {regressor_metrics['r2']:.4f}")
    
    return {
        "classifier": classifier_metrics,
        "regressor": regressor_metrics,
    }
