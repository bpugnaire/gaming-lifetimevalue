import polars as pl
from lightgbm import LGBMRegressor


def infer_cohort_regressor(
    model: LGBMRegressor, infer_df: pl.DataFrame
) -> pl.DataFrame:
    X_infer = infer_df.drop(["cohort", "user_id", "d120_rev"]).to_pandas()

    y_pred = model.predict(X_infer)

    infer_df = infer_df.with_columns(pl.Series("predicted_d120_rev", y_pred))

    return infer_df
