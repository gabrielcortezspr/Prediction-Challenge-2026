from __future__ import annotations

from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def describe_train_patterns(train_df: pd.DataFrame, target_column: str) -> str:
    null_title = int(train_df["title"].isna().sum())
    null_text = int(train_df["text"].isna().sum())
    class_dist = train_df[target_column].value_counts().sort_index().to_dict()

    return (
        f"Amostras train={len(train_df)} | "
        f"title nulo={null_title} | text nulo={null_text} | "
        f"distribuicao {target_column}={class_dist}"
    )
