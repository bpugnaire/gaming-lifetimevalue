import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import polars as pl
    from pathlib import Path
    import plotly.express as px
    from datetime import datetime, timedelta
    import numpy as np
    import plotly.graph_objects as go

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.metrics import confusion_matrix

    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    return (
        LGBMClassifier,
        Path,
        accuracy_score,
        classification_report,
        confusion_matrix,
        go,
        lgb,
        np,
        pl,
        train_test_split,
    )


@app.cell
def _(Path, pl):
    data_path = Path("data/raw/")
    train_data = pl.read_parquet(data_path / "train_samples.parquet")
    return (train_data,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Preprocessing
    """)
    return


@app.cell
def _():
    cols_to_drop = ['app_id', 'game_type', '__index_level_0__'] #ad_network_id and city could also be dropped if needed
    return (cols_to_drop,)


@app.cell
def _(pl, train_data):
    no_rev_users = train_data.filter(pl.col("d120_rev") == 0)
    rev_users = train_data.filter(pl.col("d120_rev") > 0)
    return no_rev_users, rev_users


@app.cell
def _(pl, rev_users):
    rev_users_segmented = rev_users.with_columns(
        top_rev_1pct = pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.99),
        top_rev_5pct = pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.95),
        top_rev_20pct = pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.80),
        top_rev_50pct = pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.50),
    )
    return (rev_users_segmented,)


@app.cell
def _(no_rev_users, pl, rev_users_segmented):
    rev_cohorts = rev_users_segmented.with_columns(
        cohort = pl.when(pl.col("top_rev_1pct")).then(pl.lit("Top 1%"))
                .when(pl.col("top_rev_5pct")).then(pl.lit("Top 5%"))
                .when(pl.col("top_rev_20pct")).then(pl.lit("Top 20%"))
                .when(pl.col("top_rev_50pct")).then(pl.lit("Top 50%"))
                .otherwise(pl.lit("Low Revenue"))
    )
    no_rev_cohorts = no_rev_users.with_columns(
        cohort = pl.lit("No Revenue")
    )
    return no_rev_cohorts, rev_cohorts


@app.cell
def _():
    cum_features_horizons = [f'd{dx}' for dx in [3, 7, 14, 30, 60, 90, 120]]
    cum_features_horizons
    return (cum_features_horizons,)


@app.cell
def _(cum_features_horizons, train_data):
    # all future cumulative features
    future_cum_cols = [col for col in train_data.columns if any(prefix in col for prefix in cum_features_horizons)]
    return (future_cum_cols,)


@app.cell
def _(cols_to_drop, future_cum_cols, no_rev_cohorts, rev_cohorts):
    rev_cohorts_cleaned = rev_cohorts.drop(cols_to_drop + ['top_rev_1pct', 'top_rev_5pct', 'top_rev_20pct', "top_rev_50pct"]+future_cum_cols)
    no_rev_cohorts_cleaned = no_rev_cohorts.drop(cols_to_drop + future_cum_cols)
    return no_rev_cohorts_cleaned, rev_cohorts_cleaned


@app.cell
def _(no_rev_cohorts_cleaned, pl, rev_cohorts_cleaned):
    combined_data = pl.concat([rev_cohorts_cleaned, no_rev_cohorts_cleaned])
    return (combined_data,)


@app.cell
def _(combined_data):
    combined_data.head()
    return


@app.cell
def _(combined_data):
    from skrub import TableReport

    TableReport(combined_data)
    return


@app.cell
def _(pl):
    def bin_high_cardinality(df: pl.DataFrame, column: str, top_n: int = 10) -> pl.DataFrame:
        """
        Keeps the top_n most frequent values in a categorical column and replaces 
        everything else with the string 'Other'.
        """
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
    def process_install_date(df: pl.DataFrame, date_col: str ) -> pl.DataFrame:
        """
        Extracts Year, Month, and Day of Week from a date string column 
        and drops the original column.
        """
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
        """
        Converts specified columns in the DataFrame to categorical data type.
        """
        for col in cat_cols:
            df = df.with_columns(pl.col(col).cast(pl.Categorical).to_physical().alias(col))
        return df
    return (convert_to_categorical,)


@app.cell
def _(pl):
    def add_gaming_velocity_features(df: pl.DataFrame) -> pl.DataFrame:

        return df.with_columns([
            # 1. Monetization Intensity: How much revenue per session?
            (pl.col("d0_rev") / pl.col("session_count_d0").replace(0, 1))
            .alias("rev_per_session"),

            # 2. Engagement Depth: How many games played per session?
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
def _(
    add_gaming_velocity_features,
    bin_high_cardinality,
    cat_cols,
    combined_data,
    convert_to_categorical,
    pl,
    process_install_date,
):
    def apply_transformations(df: pl.DataFrame, cat_cols: list) -> pl.DataFrame:
        df = bin_high_cardinality(df, "campaign_id", top_n=10)
        df = bin_high_cardinality(df, "model", top_n=30)
        df = bin_high_cardinality(df, "city", top_n=20)
        df = process_install_date(df, "install_date")
        df = add_gaming_velocity_features(df)
        df = convert_to_categorical(df, cat_cols)
        return df
    processed_data = apply_transformations(combined_data, cat_cols=cat_cols)
    processed_data.head()
    return (processed_data,)


@app.cell
def _(pl, processed_data):
    processed_data.filter((pl.col("d0_rev") == 0) &(pl.col("game_count_d0")==0)).group_by("cohort").len()
    return


@app.cell
def _(pl, processed_data):
    processed_data.filter((pl.col("d0_rev") == 0)&(pl.col("cohort")=='Top 1%')).head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training a classifier
    """)
    return


@app.cell
def _(processed_data):
    X = processed_data.drop(["cohort", "user_id"]).to_pandas()
    return


@app.cell
def _(
    LGBMClassifier,
    accuracy_score,
    classification_report,
    confusion_matrix,
    go,
    lgb,
    np,
    pl,
    train_test_split,
):

    def training_evaluation_pipeline(df, params, cat_cols, target_map):
        y = df.with_columns(pl.col("cohort").replace(target_map).cast(pl.Int64)).select("cohort").to_pandas()
        X = df.drop(["cohort", "user_id"]).to_pandas()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        counts = y_train['cohort'].value_counts()
        total_samples = len(y_train)
        num_classes = y_train['cohort'].nunique()
        weights_map = (total_samples / (num_classes * counts)).to_dict()

        train_weights = y_train['cohort'].map(weights_map).values.tolist()

        model = LGBMClassifier(
            n_estimators=1000,
            **params
        )
        model.fit(
            X_train, 
            y_train.values.ravel(),
            sample_weight=train_weights,
            eval_set=[(X_train, y_train.values.ravel()), (X_test, y_test.values.ravel())],
            eval_names=["train", "valid"],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        y_pred = model.predict(X_test)

        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=list(target_map.keys())))
        return y_pred, y_test, model


    def plot_confusion_matrix(y_test, y_pred, target_map):
        labels = list(target_map.keys())
        cm = confusion_matrix(y_test, y_pred)

        cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig = go.Figure(data=go.Heatmap(
            z=cm_perc,
            x=labels,
            y=labels,
            text=np.around(cm_perc, 2),
            texttemplate="%{text}",
            colorscale='Viridis',
            hoverinfo='z'
        ))

        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Cohort',
            yaxis_title='Actual Cohort',
            width=600,
            height=600
        )

        return fig
    return plot_confusion_matrix, training_evaluation_pipeline


@app.cell
def _(pl, processed_data):
    top_users = processed_data.filter(pl.col("cohort").is_in(["Top 1%", "Top 5%", "Top 20%"]))

    flop_users = processed_data.filter(pl.col("cohort").is_in(["No Revenue", "Low Revenue", "Top 50%"]))

    print(f"Top users: {len(top_users)}")
    print(f"Low revenue users: {len(flop_users)}")
    return flop_users, top_users


@app.cell
def _(flop_users, pl, processed_data, top_users):
    downsampled_flop_users = flop_users.sample(fraction=0.2, seed=42) 
    balanced_dataset = pl.concat([top_users, downsampled_flop_users])

    print(f"Original size: {len(processed_data)}")
    print(f"Balanced size: {len(balanced_dataset)}")
    return (balanced_dataset,)


@app.cell
def _():
    target_map = {
            "No Revenue": 0,
            "Low Revenue": 1,
            "Top 50%": 2,
            "Top 20%": 3,
            "Top 5%": 4,
            "Top 1%": 5,
        }

    cat_cols = [
        "platform", "country", "campaign_type", 
        "campaign_id", "model", "manufacturer", "mobile_classification", 
    ]
    return cat_cols, target_map


@app.cell
def _():
    params = {
        "objective": "multiclass",
        "num_class": 6,
        "metric": "multi_logloss",
         
    }
    return (params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Was somewhat expected but the training set is way too imbalanced. First attempts at classification had terrible results.
    The classifier fails at identifying our top customers. We introduced weights to help the model focus on the minority classes and downsampled the majority class to balance our dataset a bit better.
    """)
    return


@app.cell
def _(
    balanced_dataset,
    cat_cols,
    params,
    target_map,
    training_evaluation_pipeline,
):
    y_pred, y_test, model = training_evaluation_pipeline(balanced_dataset,target_map=target_map, cat_cols=cat_cols, params=params)
    return y_pred, y_test


@app.cell
def _(plot_confusion_matrix, target_map, y_pred, y_test):
    plot_confusion_matrix(y_test, y_pred, target_map)
    return


@app.cell
def _(plot_confusion_matrix, y_pred, y_test):
    y_test_binned = y_test['cohort'].map({0:0, 1:0, 2:0, 3:1, 4:1, 5:1})
    y_pred_binned = (y_pred >= 3).astype(int)

    binary_map = {"Low Performers": 0, "Top Performers": 1}
    plot_confusion_matrix(y_test_binned, y_pred_binned, binary_map)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    The classifier performance are not impressive but we can see that with proper balancing of the dataset and feature engineering we can start to distinguish high-value users from low-value ones.
    """)
    return


if __name__ == "__main__":
    app.run()
