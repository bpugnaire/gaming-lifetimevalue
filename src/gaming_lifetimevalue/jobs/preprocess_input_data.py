from pathlib import Path
import polars as pl

from gaming_lifetimevalue.transforms.preprocessing import (
    add_segmentation_cohorts,
    remove_columns_per_horizons,
    remove_redundant_columns,
)
from gaming_lifetimevalue.transforms.feature_engineering import (
    add_gaming_velocity_features,
    process_install_date,
    convert_to_categorical,
    bin_high_cardinality,
)


def preprocess_input_data(data_path: str, cat_cols: list[str]) -> pl.DataFrame:
    data_path = Path(data_path)
    df = pl.read_parquet(data_path)
    df = remove_redundant_columns(
        df, redundant_cols=["app_id", "game_type", "__index_level_0__"]
    )
    df = df.filter(pl.col("d120_rev").is_not_null())
    df = add_segmentation_cohorts(df)
    df = remove_columns_per_horizons(
        df, horizons=[3, 7, 14, 30, 60, 90, 120], keep_target=True
    )
    df = process_install_date(df, date_col="install_date")
    df = bin_high_cardinality(df, "campaign_id", top_n=10)
    df = bin_high_cardinality(df, "model", top_n=30)
    df = bin_high_cardinality(df, "city", top_n=20)
    df = add_gaming_velocity_features(df)
    df = convert_to_categorical(df, cat_cols=cat_cols)
    return df
