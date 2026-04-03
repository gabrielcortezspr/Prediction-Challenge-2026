"""
Ponto de entrada do desafio de classificacao multiclasse (reviews em portugues).

Orquestra: carregar CSVs, preprocessar, extrair features, treinar um pipeline
(sklearn) com texto + numeros, validar (CV opcional e holdout), gerar submission.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from features import build_feature_frame, get_numeric_feature_columns
from model import build_training_pipeline
from preprocessing import load_train_test, prepare_datasets
from utils import describe_train_patterns, evaluate_predictions, log


def infer_target_column(sample_submission_path: str) -> str:
    """
    Descobre o nome da coluna de predicao a partir do sample_submission:
    segunda coluna apos 'id' (formato tipico de competicoes).
    """
    if not Path(sample_submission_path).exists():
        raise FileNotFoundError(sample_submission_path)

    sample_cols = pd.read_csv(sample_submission_path, nrows=1).columns.tolist()
    if len(sample_cols) != 2:
        raise ValueError("sample_submission deve ter exatamente 2 colunas")
    if sample_cols[0] != "id":
        raise ValueError("A primeira coluna do sample_submission deve ser 'id'")
    return sample_cols[1]


def resolve_existing_path(candidates: list[str]) -> str:
    """Retorna o primeiro caminho da lista que existir no disco (train/test padroes)."""
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"Nenhum arquivo encontrado entre: {candidates}")


def parse_args() -> argparse.Namespace:
    """Define e le todos os argumentos de linha de comando do script."""
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
        "--target-column",
        default="rating",
        help="Coluna-alvo fallback quando sample_submission nao estiver disponivel",
    )
    parser.add_argument(
        "--submission-path",
        default="submission.csv",
        help="Caminho de saida para o arquivo de submissao",
    )
    parser.add_argument(
        "--model",
        default="linear_svm",
        choices=["linear_svm", "logreg", "lightgbm"],
        help="Modelo a ser treinado",
    )
    parser.add_argument(
        "--count-max-features",
        type=int,
        default=30000,
        help="Numero maximo de features do CountVectorizer",
    )
    parser.add_argument(
        "--use-char-ngrams",
        action="store_true",
        default=True,  # char n-grams ativados por padrao
        help="Ativa CountVectorizer de caracteres (char_wb 3-5) em paralelo ao de palavras",
    )
    parser.add_argument(
        "--char-max-features",
        type=int,
        default=20000,
        help="Numero maximo de features do vetor de caracteres",
    )
    parser.add_argument(
        "--feature-mode",
        default="full",
        choices=["full", "minimal"],
        help="Conjunto de features numericas: full (completo) ou minimal (mais enxuto)",
    )
    parser.add_argument(
        "--svm-c",
        type=float,
        default=10.0,
        help="Parametro C do LinearSVC",
    )
    parser.add_argument(
        "--svm-class-weight",
        default="balanced",
        choices=["none", "balanced"],
        help="Class weight do LinearSVC",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Numero de folds para validacao cruzada estratificada (0 desativa)",
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
    """Fluxo completo: dados -> features -> treino -> metricas -> submissao."""
    args = parse_args()

    # Resolve caminhos de train/test se o usuario nao passou explicitamente.
    train_path = args.train_path or resolve_existing_path(
        ["train_class.csv", "train.csv", "train(1).csv"]
    )
    test_path = args.test_path or resolve_existing_path(["test_class.csv", "test.csv"])
    try:
        target_column = infer_target_column(args.sample_submission_path)
        has_sample = True
    except FileNotFoundError:
        target_column = args.target_column
        has_sample = False
        log(
            "sample_submission nao encontrado; usando --target-column="
            f"{target_column} e formato padrao id,{target_column}"
        )

    log("Carregando dados...")
    train_raw, test_raw = load_train_test(train_path, test_path, target_column=target_column)

    # Resumo rapido do train (nulos, distribuicao de classes) para acompanhar o run.
    log(describe_train_patterns(train_raw, target_column=target_column))

    log("Aplicando preprocessamento...")
    train_clean, test_clean, _asin_encoder = prepare_datasets(train_raw, test_raw)

    log("Construindo features obrigatorias...")
    train_features = build_feature_frame(train_clean)
    test_features = build_feature_frame(test_clean)

    y = train_raw[target_column].astype(int)
    numeric_columns = get_numeric_feature_columns(mode=args.feature_mode)

    # Pipeline: vetoriza texto (palavras e opcionalmente chars), escala colunas numericas, classifica.
    pipeline = build_training_pipeline(
        model_name=args.model,
        max_count_features=args.count_max_features,
        use_char_ngrams=args.use_char_ngrams,
        max_char_features=args.char_max_features,
        numeric_columns=numeric_columns,
        random_state=args.random_state,
        svm_c=args.svm_c,
        svm_class_weight=args.svm_class_weight,
    )

    if args.cv_folds and args.cv_folds > 1:
        log(f"Rodando validacao cruzada estratificada com {args.cv_folds} folds...")
        cv = StratifiedKFold(
            n_splits=args.cv_folds,
            shuffle=True,
            random_state=args.random_state,
        )
        cv_result = cross_validate(
            pipeline,
            train_features,
            y,
            cv=cv,
            scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
            n_jobs=-1,
            return_train_score=False,
        )
        f1_scores = cv_result["test_f1_macro"]
        acc_scores = cv_result["test_accuracy"]
        log(
            "CV f1_macro: "
            f"{f1_scores.mean():.5f} +/- {f1_scores.std():.5f}"
        )
        log(
            "CV accuracy: "
            f"{acc_scores.mean():.5f} +/- {acc_scores.std():.5f}"
        )

    # Holdout estratificado para metricas e relatorio antes do fit final.
    X_train, X_val, y_train, y_val = train_test_split(
        train_features,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    log(f"Treinando modelo ({args.model}) com split train/val...")
    pipeline.fit(X_train, y_train)
    val_preds = pipeline.predict(X_val)

    metrics = evaluate_predictions(y_val, val_preds)
    log(f"Validation f1_macro: {metrics['f1_macro']:.5f}")
    log(f"Validation accuracy: {metrics['accuracy']:.5f}")
    log("Classification report:\n" + str(metrics["classification_report"]))
    log("Confusion matrix:\n" + str(metrics["confusion_matrix_text"]))

    # Refit em todo o train para maximizar dados na submissao.
    log("Re-treinando em 100% do train para gerar submissao...")
    pipeline.fit(train_features, y)
    test_preds = pipeline.predict(test_features)

    submission = pd.DataFrame(
        {
            "id": test_raw["id"],
            target_column: pd.Series(test_preds).astype(int),
        }
    )
    expected_cols = (
        pd.read_csv(args.sample_submission_path, nrows=1).columns.tolist()
        if has_sample
        else ["id", target_column]
    )
    submission = submission[expected_cols]
    submission.to_csv(args.submission_path, index=False)

    log(f"Arquivo de submissao salvo em: {args.submission_path}")


if __name__ == "__main__":
    main()
