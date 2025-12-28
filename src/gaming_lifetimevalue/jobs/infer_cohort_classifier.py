import polars as pl
from lightgbm import LGBMClassifier


def infer_cohort_classifier(
    model: LGBMClassifier, infer_df: pl.DataFrame, target_map: dict
) -> pl.DataFrame:
    X_infer = infer_df.drop(["cohort", "user_id", "d120_rev"]).to_pandas()

    y_pred = model.predict(X_infer)

    inverse_target_map = {v: k for k, v in target_map.items()}
    cohort_labels = [inverse_target_map[pred] for pred in y_pred]

    infer_df = infer_df.with_columns(pl.Series("predicted_cohort", cohort_labels))

    return infer_df
