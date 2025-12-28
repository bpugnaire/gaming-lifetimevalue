import polars as pl
import lightgbm as lgb

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from gaming_lifetimevalue.evaluation.metrics import evaluate_classifier


def train_cohort_classifier(
    train_df: pl.DataFrame, lgbm_params: dict, cat_cols: list[str], target_map: dict
):
    top_users = train_df.filter(pl.col("cohort").is_in(["Top 1%", "Top 5%", "Top 20%"]))
    flop_users = train_df.filter(pl.col("cohort").is_in(["Low Revenue", "Top 50%"]))

    top_count = len(top_users)
    flop_count = len(flop_users)
    sample_fraction = top_count / flop_count

    downsampled_flop_users = flop_users.sample(fraction=sample_fraction, seed=42)
    balanced_dataset = pl.concat([top_users, downsampled_flop_users])

    y = (
        balanced_dataset.with_columns(
            pl.col("cohort").replace(target_map).cast(pl.Int64)
        )
        .select("cohort")
        .to_pandas()
    )
    X = balanced_dataset.drop(["cohort", "user_id"]).to_pandas()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    class_weights = {
        0: 1.0,  # Low Revenue - baseline
        1: 2.0,  # Top 50% - 2x more important
        2: 10.0,  # Top 20% - 10x more important
        3: 50.0,  # Top 5% - 50x more important
        4: 100.0,  # Top 1% - 100x more important
    }

    train_weights = y_train["cohort"].map(class_weights).values.tolist()

    model = LGBMClassifier(
        **lgbm_params,
    )
    model.fit(
        X_train,
        y_train.values.ravel(),
        sample_weight=train_weights,
        eval_set=[(X_train, y_train.values.ravel()), (X_test, y_test.values.ravel())],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )
    y_pred = model.predict(X_test)

    metrics = evaluate_classifier(
        y_test.values.ravel(), y_pred, target_names=list(target_map.keys())
    )

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
    print("\nClassification Report:")
    for class_name, class_metrics in metrics["classification_report"].items():
        if class_name.isdigit():
            print(
                f"Class {class_name} - Precision: {class_metrics['precision']:.4f}, "
                f"Recall: {class_metrics['recall']:.4f}, F1-Score: {class_metrics['f1-score']:.4f}"
            )
    return model
