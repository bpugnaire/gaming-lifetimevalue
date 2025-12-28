# gaming-lifetimevalue

## Quick Start

```bash
git clone <repo-url>
make install
```

## Requirements

Mac with M-series CPU needs libomp for lightgbm:
```bash
brew install libomp
```

Add data samples to the `data/raw` folder.
The code currently expect a `train_samples.parquet` and a `test_samples.parquet` to work

## Exploration

To view and explore the different marimo notebooks of the project run
```bash
make notebook
```
to launch the server

## Pipelines
To prepare the training and validation datasets
```bash
make preprocessing
```

To train the LFT prediction pipeline
```bash
make training
```

To collect the results of applying the pipeline to the `test_samples.parquet` and create the predictions in 'data/predictions/'
```bash
make inference
```

## What I would do with more time

### For production robustness
- Add strong type checking and column schema to input and output dataframes (pydantic, pandera)
- Add logging to jobs and pipelines
- Implement data preprocessing in dbt for better readability and automatic lineage and quality tests

### For ML performance
- Increase training dataset size by infering `120d_rev' from recent horizons (up to horizon 30 or 60)
- Spend  more time on feature engineering (remove very low importance columns, add flag on key features like that campaign_id that is over represented in top revenue users)
- Add an hyperparameter tuning step in the training pipeline (ray tune / optuna)
- Add add model lifecylcle in MLflow registry to always use the most performant model on inference

### For adoption
- Add more metrics that focus on business KPIs (eg. '% missed whales')
- Build a dashboard for users to track performance of the model over time
