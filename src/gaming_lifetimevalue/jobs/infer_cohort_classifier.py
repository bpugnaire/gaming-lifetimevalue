import polars as pl
from lightgbm import LGBMClassifier


def infer_cohort_classifier(
    model: LGBMClassifier, infer_df: pl.DataFrame, target_map: dict
) -> pl.DataFrame:
    cols_to_drop = []
    for col in ["user_id", "d120_rev", "cohort"]:
        if col in infer_df.columns:
            cols_to_drop.append(col)

    X_infer = infer_df.drop(cols_to_drop).to_pandas()

    if X_infer.empty or X_infer.shape[1] == 0:
        raise ValueError(
            f"No features left after dropping {cols_to_drop}. Available columns: {infer_df.columns}"
        )

    y_pred = model.predict(X_infer)

    inverse_target_map = {v: k for k, v in target_map.items()}
    cohort_labels = [inverse_target_map[pred] for pred in y_pred]

    infer_df = infer_df.with_columns(pl.Series("predicted_cohort", cohort_labels))

    return infer_df
