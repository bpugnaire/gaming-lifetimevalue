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
        pd,
        pl,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Cohort-Based Regression
    Train a separate regression model for each revenue cohort we defined
    """)
    return


@app.cell
def _(Path, pl):
    data_path = Path("data/raw")
    train_data = pl.read_parquet(data_path / "train_samples.parquet").filter(
        pl.col("d120_rev").is_not_null()
    )

    print(f"Train shape: {train_data.shape}")
    return (train_data,)


app._unparsable_cell(
    r"""
    train_data.select('install_date').max().
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Preprocessing
    """)
    return


@app.cell
def _():
    cols_to_drop = ['app_id', 'game_type', '__index_level_0__']
    return (cols_to_drop,)


@app.cell
def _(pl, train_data):
    no_rev_users = train_data.filter(pl.col("d120_rev") == 0)
    rev_users = train_data.filter(pl.col("d120_rev") > 0)

    print(f"No revenue users: {len(no_rev_users)}")
    print(f"Revenue users: {len(rev_users)}")
    return (rev_users,)


@app.cell
def _(pl, rev_users):
    rev_users_segmented = rev_users.with_columns(
        top_rev_1pct=pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.99),
        top_rev_5pct=pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.95),
        top_rev_20pct=pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.80),
        top_rev_50pct=pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.50),
    )
    return (rev_users_segmented,)


@app.cell
def _(pl, rev_users_segmented):
    rev_cohorts = rev_users_segmented.with_columns(
        cohort=pl.when(pl.col("top_rev_1pct"))
        .then(pl.lit("Top 1%"))
        .when(pl.col("top_rev_5pct"))
        .then(pl.lit("Top 5%"))
        .when(pl.col("top_rev_20pct"))
        .then(pl.lit("Top 20%"))
        .when(pl.col("top_rev_50pct"))
        .then(pl.lit("Top 50%"))
        .otherwise(pl.lit("Low Revenue"))
    )

    rev_cohorts.group_by("cohort").agg(pl.len().alias("len"))
    return (rev_cohorts,)


@app.cell
def _():
    cum_features_horizons = [f'd{dx}' for dx in [3, 7, 14, 30, 60, 90, 120]]
    return (cum_features_horizons,)


@app.cell
def _(cum_features_horizons, train_data):
    future_cum_cols = [
        col for col in train_data.columns 
        if any(prefix in col for prefix in cum_features_horizons)
    ]
    future_cum_cols_without_target = [col for col in future_cum_cols if col != 'd120_rev']
    return (future_cum_cols_without_target,)


@app.cell
def _(cols_to_drop, future_cum_cols_without_target, rev_cohorts):
    rev_cohorts_cleaned = rev_cohorts.drop(
        cols_to_drop + ['top_rev_1pct', 'top_rev_5pct', 'top_rev_20pct', "top_rev_50pct"] + future_cum_cols_without_target
    )
    return (rev_cohorts_cleaned,)


@app.cell
def _(pl):
    def bin_high_cardinality(df: pl.DataFrame, column: str, top_n: int = 10) -> pl.DataFrame:
        top_values = (
            df[column]
            .value_counts()
            .sort("count", descending=True)
            .head(top_n)[column]
            .to_list()
        )

        df = df.with_columns(
            pl.when(pl.col(column).is_in(top_values))
            .then(pl.col(column))
            .otherwise(pl.lit("Other"))
            .alias(column)
        )
        return df.with_columns(
            pl.col(column).cast(pl.Categorical).to_physical().alias(column)
        )
    return (bin_high_cardinality,)


@app.cell
def _(pl):
    def process_install_date(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
        return (
            df.with_columns([
                pl.col(date_col).dt.year().alias(f"{date_col}_year"),
                pl.col(date_col).dt.month().alias(f"{date_col}_month"),
                pl.col(date_col).dt.weekday().alias(f"{date_col}_dow"),
            ])
            .drop(date_col)
        )
    return (process_install_date,)


@app.cell
def _(pl):
    def convert_to_categorical(df: pl.DataFrame, cat_cols: list) -> pl.DataFrame:
        for col in cat_cols:
            df = df.with_columns(pl.col(col).cast(pl.Categorical).to_physical().alias(col))
        return df
    return (convert_to_categorical,)


@app.cell
def _(pl):
    def add_gaming_velocity_features(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            (pl.col("d0_rev") / pl.col("session_count_d0").replace(0, 1))
            .alias("rev_per_session"),

            (pl.col("game_count_d0") / pl.col("session_count_d0").replace(0, 1))
            .alias("games_per_session"),

            (pl.col("session_length_d0") / pl.col("session_count_d0").replace(0, 1))
            .alias("avg_session_duration"),

            (pl.col("coins_spend_sum_d0") / pl.col("session_count_d0").replace(0, 1))
            .alias("spending_pressure"),

            (pl.col("rv_shown_count_d0") / pl.col("game_count_d0").replace(0, 1))
            .alias("rv_per_game"),

            (pl.col("current_level_d0") / pl.col("session_length_d0").replace(0, 1))
            .alias("level_velocity"),
        ]).fill_nan(0).fill_null(0)
    return (add_gaming_velocity_features,)


@app.cell
def _():
    cat_cols = [
        "platform", "country", "campaign_type", 
        "campaign_id", "model", "manufacturer", "mobile_classification", 
    ]
    return (cat_cols,)


@app.cell
def _(
    add_gaming_velocity_features,
    bin_high_cardinality,
    cat_cols,
    convert_to_categorical,
    pl,
    process_install_date,
    rev_cohorts_cleaned,
):
    def apply_transformations(df: pl.DataFrame, cat_cols: list) -> pl.DataFrame:
        df = bin_high_cardinality(df, "campaign_id", top_n=10)
        df = bin_high_cardinality(df, "model", top_n=30)
        df = bin_high_cardinality(df, "city", top_n=20)
        df = process_install_date(df, "install_date")
        df = add_gaming_velocity_features(df)
        df = convert_to_categorical(df, cat_cols)
        return df

    processed_data = apply_transformations(rev_cohorts_cleaned, cat_cols=cat_cols)
    processed_data.head()
    return (processed_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Train Cohort-Specific Models
    """)
    return


@app.cell
def _(lgb, mean_absolute_error, mean_squared_error, np, pl, train_test_split):
    def train_cohort_model(cohort_df: pl.DataFrame, cohort_name: str, cat_cols: list):

        X = cohort_df.drop(["cohort", "user_id", "d120_rev"]).to_pandas()
        y = cohort_df.select("d120_rev").to_pandas()["d120_rev"]

        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].fillna('missing').astype('category')

        numeric_cols = X.select_dtypes(include=['number']).columns
        X[numeric_cols] = X[numeric_cols].fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbose=-1,
            enable_categorical=True,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"\n{cohort_name} Cohort:")
        print(f"  Samples: {len(cohort_df)}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Test Mean: {y_test.mean():.4f}")
        print(f"  Pred Mean: {y_pred.mean():.4f}")

        return {
            "cohort": cohort_name,
            "model": model,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "test_mean": round(y_test.mean(), 4),
            "pred_mean": round(y_pred.mean(), 4),
        }
    return (train_cohort_model,)


@app.cell
def _(cat_cols, processed_data, train_cohort_model):
    cohort_models = {}
    cohort_results = []

    cohorts = ["Top 1%", "Top 5%", "Top 20%", "Top 50%", "Low Revenue"]

    for cohort_name in cohorts:
        cohort_data = processed_data.filter(processed_data["cohort"] == cohort_name)

        if len(cohort_data) > 0:
            result = train_cohort_model(cohort_data, cohort_name, cat_cols)
            cohort_models[cohort_name] = result["model"]
            cohort_results.append(result)
    return cohort_models, cohort_results


@app.cell
def _(cohort_results, pl):
    pl.DataFrame(cohort_results).drop(["model"])
    return


@app.cell
def _(cat_cols, cohort_models, pl, processed_data):
    def predict_with_cohort_models(df: pl.DataFrame, cohort_models: dict, cat_cols: list):
        """
        Predict using cohort-specific models.
        Assumes df has a 'cohort' column indicating which model to use.
        """
        predictions = []
        user_ids = []

        for cohort_name, model in cohort_models.items():
            cohort_data = df.filter(df["cohort"] == cohort_name)

            if len(cohort_data) > 0:
                X = cohort_data.drop(["cohort", "user_id", "d120_rev"]).to_pandas()

                # Handle missing values
                for col in cat_cols:
                    if col in X.columns:
                        X[col] = X[col].fillna('missing').astype('category')

                numeric_cols = X.select_dtypes(include=['number']).columns
                X[numeric_cols] = X[numeric_cols].fillna(0)

                pred = model.predict(X)
                predictions.extend(pred)
                user_ids.extend(cohort_data["user_id"].to_list())

        return pl.DataFrame({
            "user_id": user_ids,
            "predicted_d120_rev": predictions
        })

    sample_predictions = predict_with_cohort_models(processed_data.sample(5000), cohort_models, cat_cols)
    sample_predictions.head()
    return


@app.cell
def _(cohort_models):
    cohort_models['Top 1%'].feature_importances_
    return


@app.cell
def _(cohort_models):
    cohort_models['Top 1%'].feature_name_
    return


@app.cell
def _(cohort_models, pd):
    import plotly.express as px

    feat_names = cohort_models['Top 1%'].feature_name_
    feat_importances = cohort_models['Top 1%'].feature_importances_

    df_importance = pd.DataFrame({
        'Feature': feat_names,
        'Importance': feat_importances
    }).sort_values(by='Importance', ascending=True)

    fig = px.bar(
        df_importance.tail(20), 
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 20 Feature Importances - Top 1% Cohort',
        template='plotly_white',
        color='Importance',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        xaxis_title='Importance Score (Gain/Split)',
        yaxis_title='',
        width=800,
        height=600
    )

    fig.show()
    return df_importance, px


@app.cell
def _(df_importance, px):
    fig2 = px.bar(
        df_importance.head(10), 
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 20 Feature Importances - Top 1% Cohort',
        template='plotly_white',
        color='Importance',
        color_continuous_scale='Viridis'
    )

    fig2.update_layout(
        xaxis_title='Importance Score (Gain/Split)',
        yaxis_title='',
        width=800,
        height=600
    )

    fig2.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
