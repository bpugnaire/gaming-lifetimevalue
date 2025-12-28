import polars as pl
import lightgbm as lgb

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def train_cohort_classifier(
    train_df: pl.DataFrame, lgbm_params: dict, cat_cols: list[str], target_map: dict
):
    y = (
        train_df.with_columns(pl.col("cohort").replace(target_map).cast(pl.Int64))
        .select("cohort")
        .to_pandas()
    )
    X = train_df.drop(["cohort", "user_id"]).to_pandas()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    counts = y_train["cohort"].value_counts()
    total_samples = len(y_train)
    num_classes = y_train["cohort"].nunique()
    weights_map = (total_samples / (num_classes * counts)).to_dict()

    train_weights = y_train["cohort"].map(weights_map).values.tolist()

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

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=list(target_map.keys())))
    return model
