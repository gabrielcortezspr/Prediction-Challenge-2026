from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from preprocessing import get_portuguese_stopwords, normalize_text_for_vectorizer


def build_estimator(model_name: str, random_state: int):
    model_name = model_name.lower()

    if model_name == "linear_svm":
        return LinearSVC(C=1.0, random_state=random_state)

    if model_name == "logreg":
        return LogisticRegression(
            max_iter=1500,
            solver="lbfgs",
            random_state=random_state,
        )

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "LightGBM nao esta instalado. Rode: pip install lightgbm"
            ) from exc

        return LGBMClassifier(
            objective="multiclass",
            num_class=5,
            n_estimators=350,
            learning_rate=0.05,
            random_state=random_state,
        )

    raise ValueError("model_name deve ser 'linear_svm', 'logreg' ou 'lightgbm'")


def build_training_pipeline(
    model_name: str,
    max_count_features: int,
    numeric_columns: list[str],
    random_state: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "count_vectorizer",
                CountVectorizer(
                    preprocessor=normalize_text_for_vectorizer,
                    stop_words=get_portuguese_stopwords(),
                    ngram_range=(1, 2),
                    max_features=max_count_features,
                    strip_accents="unicode",
                    lowercase=True,
                ),
                "combined_text",
            ),
            (
                "numeric",
                StandardScaler(with_mean=False),
                numeric_columns,
            ),
        ],
        remainder="drop",
    )

    estimator = build_estimator(model_name=model_name, random_state=random_state)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
