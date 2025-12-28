import polars as pl
from lightgbm import LGBMRegressor


def infer_cohort_regressor(
    model: LGBMRegressor, infer_df: pl.DataFrame, cat_cols: list
) -> pl.DataFrame:
    cols_to_drop = []
    for col in ["cohort", "predicted_cohort", "user_id", "d120_rev"]:
        if col in infer_df.columns:
            cols_to_drop.append(col)
    
    X_infer = infer_df.drop(cols_to_drop).to_pandas()
    
    for col in cat_cols:
        if col in X_infer.columns:
            X_infer[col] = X_infer[col].fillna("missing").astype("category")
    
    numeric_cols = X_infer.select_dtypes(include=["number"]).columns
    X_infer[numeric_cols] = X_infer[numeric_cols].fillna(0)

    y_pred = model.predict(X_infer)

    infer_df = infer_df.with_columns(pl.Series("predicted_d120_rev", y_pred))

    return infer_df
