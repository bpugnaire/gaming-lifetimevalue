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
        top_rev_1pct = pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.99),
        top_rev_5pct = pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.95),
        top_rev_20pct = pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.80),
        top_rev_50pct = pl.col("d120_rev") >= pl.col("d120_rev").quantile(0.50),
    )
    rev_with_cohorts = rev_users_segmented.with_columns(
        cohort = pl.when(pl.col("top_rev_1pct")).then(pl.lit("Top 1%"))
                .when(pl.col("top_rev_5pct")).then(pl.lit("Top 5%"))
                .when(pl.col("top_rev_20pct")).then(pl.lit("Top 20%"))
                .when(pl.col("top_rev_50pct")).then(pl.lit("Top 50%"))
                .otherwise(pl.lit("Others"))
    )
    no_rev_with_cohorts = no_rev_users.with_columns(
        cohort = pl.lit("No Revenue")
    )
    rev_with_cohorts = rev_with_cohorts.drop(['top_rev_1pct', 'top_rev_5pct', 'top_rev_20pct', "top_rev_50pct"])

    return pl.concat([no_rev_with_cohorts, rev_with_cohorts])


def remove_columns_per_horizons(pl_df: pl.DataFrame, horizons: list[int]) -> pl.DataFrame:
    """
    Remove all columns from the DataFrame that are associated with the specified horizons.
    Args:
        pl_df (pl.DataFrame): Input DataFrame.
        horizons (list[int]): List of horizon values.

    Returns:
        pl.DataFrame: DataFrame with specified horizons columns removed.
    """
    horizons_cols = [col for col in pl_df.columns if any(horizon in col for horizon in horizons)]
    existing_cols_to_remove = [col for col in horizons_cols if col in pl_df.columns]
    return pl_df.drop(existing_cols_to_remove)

def remove_redundant_columns(pl_df: pl.DataFrame, redundant_cols: list[str]) -> pl.DataFrame:
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

def bin_high_cardinality(pl_df: pl.DataFrame, column: str, top_n: int = 10) -> pl.DataFrame:
    """
    Bin high-cardinality categorical variable into top N categories and 'Other'.
    Args:
        pl_df (pl.DataFrame): Input DataFrame.
        column (str): Column name to bin.
        top_n (int): Number of top categories to retain.

    Returns:
        pl.DataFrame: DataFrame with binned categorical variable.
    """
    top_values = (
        pl_df[column]
        .value_counts()
        .sort("count", descending=True)
        .head(top_n)[column]
        .to_list()
    )

    pl_df = pl_df.with_columns(
        pl.when(pl.col(column).is_in(top_values))
        .then(pl.col(column))
        .otherwise(pl.lit("Other"))
        .alias(column)
    )
    return pl_df.with_columns(
        pl.col(column).cast(pl.Categorical).to_physical().alias(column)
    )

def process_install_date(pl_df: pl.DataFrame, date_col: str ) -> pl.DataFrame:
    """
    Process install date column to extract useful features.
    Args:
        pl_df (pl.DataFrame): Input DataFrame.
        date_col (str): Name of the install date column.

    Returns:
        pl.DataFrame: DataFrame with processed install date features.
    """
    return (
        pl_df.with_columns([
            pl.col(date_col).dt.year().alias(f"{date_col}_year"),
            pl.col(date_col).dt.month().alias(f"{date_col}_month"),
            pl.col(date_col).dt.weekday().alias(f"{date_col}_dow"),
        ])
        .drop(date_col)
    )

def convert_to_categorical(pl_df: pl.DataFrame, cat_cols: list) -> pl.DataFrame:
    """
    Convert specified columns to categorical data type.
    Args:
        pl_df (pl.DataFrame): Input DataFrame.
        cat_cols (list): List of column names to convert.
    Returns:
        pl.DataFrame: DataFrame with specified columns converted to categorical.
    """
    for col in cat_cols:
        pl_df = pl_df.with_columns(pl.col(col).cast(pl.Categorical).to_physical().alias(col))
    return pl_df


def add_gaming_velocity_features(pl_df: pl.DataFrame) -> pl.DataFrame:
    """
    Creates intensity and velocity features to help distinguish high-value users.
    Handles potential division by zero using 'fill_nan' and 'fill_null'.
    Args:
        pl_df (pl.DataFrame): Input DataFrame with gaming metrics.
    Returns:
        pl.DataFrame: DataFrame with added gaming velocity features.
    """
    return pl_df.with_columns([
        (pl.col("d0_rev") / pl.col("session_count_d0").replace(0, 1))
        .alias("rev_per_session"),
        
        (pl.col("game_count_d0") / pl.col("session_count_d0").replace(0, 1))
        .alias("games_per_session"),
        
        (pl.col("session_length_d0") / pl.col("session_count_d0").replace(0, 1))
        .alias("avg_session_duration"),

        (pl.col("coins_spend_sum_d0") / pl.col("session_count_d0").replace(0, 1))
        .alias("coins_spent_per_session"),
        
        (pl.col("rv_shown_count_d0") / pl.col("game_count_d0").replace(0, 1))
        .alias("rv_per_game"),
        
        (pl.col("current_level_d0") / pl.col("session_length_d0").replace(0, 1))
        .alias("level_velocity"),
        
    ]).fill_nan(0).fill_null(0)