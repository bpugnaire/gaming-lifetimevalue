import polars as pl
from pathlib import Path
from datetime import timedelta
from gaming_lifetimevalue.jobs.preprocess_input_data import preprocess_input_data
from gaming_lifetimevalue.utils.config_loader import load_config


def main():
    print("Starting preprocessing pipeline...")
    # load parameters from confs/params.yml
    params = load_config()

    print("Loading raw training data...")
    raw_dataset = pl.read_parquet(params["raw_training_data_path"]).filter(
        pl.col("d120_rev").is_not_null()
    )
    # split dataset to extract last month install_date for valid set
    max_date = raw_dataset.select(pl.col("install_date").max()).item()
    cutoff_date = max_date - timedelta(days=30)
    train_dataset = raw_dataset.filter(pl.col("install_date") < cutoff_date)
    valid_dataset = raw_dataset.filter(pl.col("install_date") >= cutoff_date)

    print("Processing train and validation dataset...")
    processed_train = preprocess_input_data(
        pl_df=train_dataset, cat_cols=params["categorical_columns"]
    )
    processed_valid = preprocess_input_data(
        pl_df=valid_dataset, cat_cols=params["categorical_columns"]
    )
    print("Saving processed datasets...")
    # save dataset to the data/processed folder
    processed_train.write_parquet(
        Path(params["processed_data_path"]) / "train_data.parquet"
    )
    processed_valid.write_parquet(
        Path(params["processed_data_path"]) / "valid_data.parquet"
    )
    print("Processed datasets saved in data/processed folder.")
    print("Preprocessing pipeline completed.")


if __name__ == "__main__":
    main()
