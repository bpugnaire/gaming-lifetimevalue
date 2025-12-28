import polars as pl
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_cohort_regressor(
    train_df: pl.DataFrame, lgbm_params: dict, cohort_name: str, cat_cols: list
):
    X = train_df.drop(["cohort", "user_id", "d120_rev"]).to_pandas()
    y = train_df.select("d120_rev").to_pandas()["d120_rev"]

    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].fillna("missing").astype("category")

    numeric_cols = X.select_dtypes(include=["number"]).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    model = lgb.LGBMRegressor(
        **lgbm_params,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n{cohort_name} Cohort:")
    print(f"  Samples: {len(train_df)}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Test Mean: {y_test.mean():.4f}")
    print(f"  Pred Mean: {y_pred.mean():.4f}")

    return model
