from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from features import build_feature_frame, get_numeric_feature_columns
from model import build_training_pipeline
from preprocessing import load_train_test, prepare_datasets
from utils import describe_train_patterns, evaluate_predictions, log


def infer_target_column(sample_submission_path: str) -> str:
    sample_cols = pd.read_csv(sample_submission_path, nrows=1).columns.tolist()
    if len(sample_cols) != 2:
        raise ValueError("sample_submission deve ter exatamente 2 colunas")
    if sample_cols[0] != "id":
        raise ValueError("A primeira coluna do sample_submission deve ser 'id'")
    return sample_cols[1]


def resolve_existing_path(candidates: list[str]) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"Nenhum arquivo encontrado entre: {candidates}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de classificacao multiclasse para reviews em portugues"
    )
    parser.add_argument(
        "--train-path",
        default=None,
        help="Caminho do arquivo de treino (ex.: train_class.csv ou train.csv)",
    )
    parser.add_argument(
        "--test-path",
        default=None,
        help="Caminho do arquivo de teste (ex.: test_class.csv ou test.csv)",
    )
    parser.add_argument(
        "--sample-submission-path",
        default="sample_submission.csv",
        help="Arquivo sample_submission para inferir coluna-alvo e formato final",
    )
    parser.add_argument(
        "--submission-path",
        default="submission.csv",
        help="Caminho de saida para o arquivo de submissao",
    )
    parser.add_argument(
        "--model",
        default="logreg",
        choices=["logreg", "lightgbm"],
        help="Modelo a ser treinado",
    )
    parser.add_argument(
        "--tfidf-max-features",
        type=int,
        default=50000,
        help="Numero maximo de features do TF-IDF",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fracao para validacao holdout",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed de reproducibilidade",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_path = args.train_path or resolve_existing_path(
        ["train_class.csv", "train.csv", "train(1).csv"]
    )
    test_path = args.test_path or resolve_existing_path(["test_class.csv", "test.csv"])
    target_column = infer_target_column(args.sample_submission_path)

    log("Carregando dados...")
    train_raw, test_raw = load_train_test(train_path, test_path, target_column=target_column)

    # Leitura de padrao observado no dataset para facilitar monitoramento.
    log(describe_train_patterns(train_raw, target_column=target_column))

    log("Aplicando preprocessamento...")
    train_clean, test_clean, _asin_encoder = prepare_datasets(train_raw, test_raw)

    log("Construindo features obrigatorias...")
    train_features = build_feature_frame(train_clean)
    test_features = build_feature_frame(test_clean)

    y = train_raw[target_column].astype(int)
    numeric_columns = get_numeric_feature_columns()

    X_train, X_val, y_train, y_val = train_test_split(
        train_features,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    log(f"Treinando modelo ({args.model}) com split train/val...")
    pipeline = build_training_pipeline(
        model_name=args.model,
        max_tfidf_features=args.tfidf_max_features,
        numeric_columns=numeric_columns,
        random_state=args.random_state,
    )

    pipeline.fit(X_train, y_train)
    val_preds = pipeline.predict(X_val)

    metrics = evaluate_predictions(y_val, val_preds)
    log(f"Validation f1_macro: {metrics['f1_macro']:.5f}")
    log(f"Validation accuracy: {metrics['accuracy']:.5f}")

    log("Re-treinando em 100% do train para gerar submissao...")
    pipeline.fit(train_features, y)
    test_preds = pipeline.predict(test_features)

    submission = pd.DataFrame(
        {
            "id": test_raw["id"],
            target_column: pd.Series(test_preds).astype(int),
        }
    )
    expected_cols = pd.read_csv(args.sample_submission_path, nrows=1).columns.tolist()
    submission = submission[expected_cols]
    submission.to_csv(args.submission_path, index=False)

    log(f"Arquivo de submissao salvo em: {args.submission_path}")


if __name__ == "__main__":
    main()
