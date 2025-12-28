import polars as pl


def add_segmentation_cohorts(pl_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add segmentation cohorts to the DataFrame based on revenue quantiles.
    Args:
        pl_df (pl.DataFrame): Input DataFrame containing revenue data.

    Returns:
        pl.DataFrame: DataFrame with added segmentation cohorts.
    """
    no_rev_users = pl_df.filter(pl.col("d120_rev") == 0)
    rev_users = pl_df.filter(pl.col("d120_rev") > 0)
    rev_users_segmented = rev_users.with_columns(
        top_rev_1pct=pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.99),
        top_rev_5pct=pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.95),
        top_rev_20pct=pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.80),
        top_rev_50pct=pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.50),
    )
    rev_with_cohorts = rev_users_segmented.with_columns(
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
    no_rev_with_cohorts = no_rev_users.with_columns(cohort=pl.lit("No Revenue"))
    rev_with_cohorts = rev_with_cohorts.drop(
        ["top_rev_1pct", "top_rev_5pct", "top_rev_20pct", "top_rev_50pct"]
    )

    return pl.concat([no_rev_with_cohorts, rev_with_cohorts])


def remove_columns_per_horizons(
    pl_df: pl.DataFrame, horizons: list[int], keep_target: bool = True
) -> pl.DataFrame:
    """
    Remove all columns from the DataFrame that are associated with the specified horizons.
    Args:
        pl_df (pl.DataFrame): Input DataFrame.
        horizons (list[int]): List of horizon values.
        keep_target (bool): Whether to keep the target column.

    Returns:
        pl.DataFrame: DataFrame with specified horizons columns removed.
    """
    cum_features_horizons = [f'd{dx}' for dx in horizons]
    horizons_cols = [
        col for col in pl_df.columns if any(horizon_col in col for horizon_col in cum_features_horizons)
    ]
    existing_cols_to_remove = [col for col in horizons_cols if col in pl_df.columns]
    if keep_target:
        existing_cols_to_remove = [
            col for col in existing_cols_to_remove if col != "d120_rev"
        ]
    return pl_df.drop(existing_cols_to_remove)


def remove_redundant_columns(
    pl_df: pl.DataFrame, redundant_cols: list[str]
) -> pl.DataFrame:
    """
    Remove redundant columns from the DataFrame.
    Args:
        pl_df (pl.DataFrame): Input DataFrame.
        redundant_cols (list[str]): List of redundant column names to remove.

    Returns:
        pl.DataFrame: DataFrame with redundant columns removed.
    """
    existing_cols_to_remove = [col for col in redundant_cols if col in pl_df.columns]
    return pl_df.drop(existing_cols_to_remove)
