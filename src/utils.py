"""
Utilitarios de logging e avaliacao do modelo.

- log: impressao com timestamp.
- evaluate_predictions: accuracy, F1 macro, classification report, matriz de confusao.
- describe_train_patterns: estatisticas descritivas rapidas do conjunto de treino.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def log(message: str) -> None:
    """Imprime uma mensagem no stdout com data/hora local."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, object]:
    """
    Calcula metricas de classificacao multiclasse e formata matriz de confusao
    com rotulos real_* / pred_* para leitura em texto.
    """
    labels = sorted(pd.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"real_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "classification_report": classification_report(y_true, y_pred, digits=4),
        "confusion_matrix_text": cm_df.to_string(),
    }


def describe_train_patterns(train_df: pd.DataFrame, target_column: str) -> str:
    """
    Retorna string resumindo: tamanho do train, contagens de nulos em title/text,
    e distribuicao de classes na coluna-alvo.
    """
    null_title = int(train_df["title"].isna().sum())
    null_text = int(train_df["text"].isna().sum())
    class_dist = train_df[target_column].value_counts().sort_index().to_dict()

    return (
        f"Amostras train={len(train_df)} | "
        f"title nulo={null_title} | text nulo={null_text} | "
        f"distribuicao {target_column}={class_dist}"
    )
