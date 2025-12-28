import polars as pl

def add_gaming_velocity_features(pl_df: pl.DataFrame) -> pl.DataFrame:
    """
    Creates intensity and velocity features to help distinguish high-value users.
    Handles potential division by zero using 'fill_nan' and 'fill_null'.
    Args:
        pl_df (pl.DataFrame): Input DataFrame with gaming metrics.
    Returns:
        pl.DataFrame: DataFrame with added gaming velocity features.
    """
    return (
        pl_df.with_columns(
            [
                (pl.col("d0_rev") / pl.col("session_count_d0").replace(0, 1)).alias(
                    "rev_per_session"
                ),
                (
                    pl.col("game_count_d0") / pl.col("session_count_d0").replace(0, 1)
                ).alias("games_per_session"),
                (
                    pl.col("session_length_d0")
                    / pl.col("session_count_d0").replace(0, 1)
                ).alias("avg_session_duration"),
                (
                    pl.col("coins_spend_sum_d0")
                    / pl.col("session_count_d0").replace(0, 1)
                ).alias("coins_spent_per_session"),
                (
                    pl.col("rv_shown_count_d0") / pl.col("game_count_d0").replace(0, 1)
                ).alias("rv_per_game"),
                (
                    pl.col("current_level_d0")
                    / pl.col("session_length_d0").replace(0, 1)
                ).alias("level_velocity"),
            ]
        )
        .fill_nan(0)
        .fill_null(0)
    )

def process_install_date(pl_df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    """
    Process install date column to extract useful features.
    Args:
        pl_df (pl.DataFrame): Input DataFrame.
        date_col (str): Name of the install date column.

    Returns:
        pl.DataFrame: DataFrame with processed install date features.
    """
    return pl_df.with_columns(
        [
            pl.col(date_col).dt.year().alias(f"{date_col}_year"),
            pl.col(date_col).dt.month().alias(f"{date_col}_month"),
            pl.col(date_col).dt.weekday().alias(f"{date_col}_dow"),
        ]
    ).drop(date_col)

def convert_to_categorical(pl_df: pl.DataFrame, cat_cols: list) -> pl.DataFrame:
    """
    Convert specified features to categorical data type.
    Args:
        pl_df (pl.DataFrame): Input DataFrame.
        cat_cols (list): List of feature names to convert.
    Returns:
        pl.DataFrame: DataFrame with specified features converted to categorical.
    """
    for col in cat_cols:
        pl_df = pl_df.with_columns(
            pl.col(col).cast(pl.Categorical).to_physical().alias(col)
        )
    return pl_df

def bin_high_cardinality(
    pl_df: pl.DataFrame, column: str, top_n: int = 10
) -> pl.DataFrame:
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