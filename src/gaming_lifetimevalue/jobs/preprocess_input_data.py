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


def preprocess_input_data(
    pl_df: pl.DataFrame, cat_cols: list[str], inference: bool = False
) -> pl.DataFrame:
    pl_df = remove_redundant_columns(
        pl_df, redundant_cols=["app_id", "game_type", "__index_level_0__"]
    )

    if not inference:
        pl_df = pl_df.filter(pl.col("d120_rev").is_not_null())
        pl_df = add_segmentation_cohorts(pl_df)

    pl_df = remove_columns_per_horizons(
        pl_df, horizons=[3, 7, 14, 30, 60, 90, 120], keep_target=not inference
    )
    pl_df = process_install_date(pl_df, date_col="install_date")
    pl_df = bin_high_cardinality(pl_df, "campaign_id", top_n=10)
    pl_df = bin_high_cardinality(pl_df, "model", top_n=30)
    pl_df = bin_high_cardinality(pl_df, "city", top_n=20)
    pl_df = add_gaming_velocity_features(pl_df)
    pl_df = convert_to_categorical(pl_df, cat_cols=cat_cols)
    return pl_df
