import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    from pathlib import Path
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    return (
        Path,
        lgb,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pl,
        r2_score,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Simple Regression Baseline
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Load Data
    """)
    return


@app.cell
def _(Path, pl):
    data_path = Path("data/raw")
    train_data = pl.read_parquet(data_path / "train_samples.parquet").filter(pl.col("d120_rev").is_not_null())

    print(f"Train shape: {train_data.shape}")
    return (train_data,)


@app.cell
def _(train_data):
    train_data.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Preprocessing
    """)
    return


@app.cell
def _(train_data):
    target = "d120_rev"

    cum_features_horizons = [f'd{dx}' for dx in [3, 7, 14, 30, 60, 90, 120]]
    future_cum_cols = [col for col in train_data.columns if any(prefix in col for prefix in cum_features_horizons)]

    drop_cols = ["install_date", "user_id", 'app_id', 'game_type', '__index_level_0__', "ad_network_id", "city"] + future_cum_cols

    feature_cols = [col for col in train_data.columns if col not in drop_cols]

    print(f"Nb features: {len(feature_cols)}")
    print(feature_cols)
    return feature_cols, target


@app.cell
def _():
    cat_cols = [
        "platform", "country", "campaign_type", 
        "campaign_id", "model", "manufacturer", "mobile_classification", 
    ]
    return (cat_cols,)


@app.cell
def _(cat_cols, feature_cols, target, train_data):
    X = train_data.select(feature_cols).to_pandas()
    y = train_data.select(target).to_pandas()[target]

    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].fillna('missing').astype('category')

    numeric_cols = X.select_dtypes(include=['number']).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)

    print(f"Missing values: {X.isna().sum().sum()}")
    return X, y


@app.cell
def _(X):
    X.head()
    return


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Train Model
    """)
    return


@app.cell
def _(X_test, X_train, cat_cols, lgb, y_test, y_train):
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=-1,
        enable_categorical=True,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=False),
        ]
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluation
    """)
    return


@app.cell
def _(y_test):
    y_test
    return


@app.cell
def _(
    X_test,
    mean_absolute_error,
    mean_squared_error,
    model,
    np,
    r2_score,
    y_test,
):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    return (y_pred,)


@app.cell
def _(y_test):
    y_test.index
    return


@app.cell
def _(pl, y_test):
    y_test_pl = pl.from_pandas(y_test)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Segmented metrics
    """)
    return


@app.cell
def _(
    mean_absolute_error,
    mean_squared_error,
    np,
    pl,
    r2_score,
    y_pred,
    y_test,
):
    from gaming_lifetimevalue.transforms.preprocessing import add_segmentation_cohorts

    df = pl.DataFrame({
        "d120_rev": y_test.values,
        "y_pred": y_pred
    })

    df_with_cohorts = add_segmentation_cohorts(df)

    def compute_metrics_segmented(df_with_cohorts):
        results = []
        for cohort_name, group in df_with_cohorts.group_by("cohort"):
            y_true_seg = group["d120_rev"].to_numpy()
            y_pred_seg = group["y_pred"].to_numpy()

            mae = mean_absolute_error(y_true_seg, y_pred_seg)
            rmse = np.sqrt(mean_squared_error(y_true_seg, y_pred_seg))

            try:
                r2 = r2_score(y_true_seg, y_pred_seg)
            except:
                r2 = float('nan')

            results.append({
                "Cohort": cohort_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Count": len(group),
                "Test mean": y_true_seg.mean(),
                "Pred mean": y_pred_seg.mean(),
            })

        return pl.DataFrame(results).sort("MAE", descending=True)
    compute_metrics_segmented(df_with_cohorts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate Test Predictions
    """)
    return


@app.cell
def _():
    # X_test = test_df.select(feature_cols).to_pandas()
    # X_test = X_test.fillna(0)

    # test_predictions = model.predict(X_test)

    # submission = pl.DataFrame({
    #     "user_id": test_df["user_id"],
    #     "d120_rev": test_predictions
    # })

    # submission.head()
    return


@app.cell
def _():
    # # Save predictions
    # output_path = Path("../data/predictions")
    # output_path.mkdir(parents=True, exist_ok=True)

    # submission.write_csv(output_path / "baseline_predictions.csv")
    # print(f"Predictions saved to {output_path / 'baseline_predictions.csv'}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
