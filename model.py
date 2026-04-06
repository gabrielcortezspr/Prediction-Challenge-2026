"""
Montagem do Pipeline sklearn: vetorizacao de texto + escalonamento numerico + classificador.

Suporta LinearSVC, Regressao Logistica e (opcional) LightGBM. O preprocessador usa
ColumnTransformer para aplicar CountVectorizer em combined_text e StandardScaler
nas colunas numericas vindas de features.py.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from preprocessing import get_portuguese_stopwords, normalize_text_for_vectorizer


def build_estimator(
    model_name: str,
    random_state: int,
    svm_c: float = 1.0,
    svm_class_weight: str | None = None,
):
    """
    Instancia o classificador final conforme model_name.
    LinearSVC usa dual=False (recomendado quando n_samples > n_features com bag-of-words).
    """
    model_name = model_name.lower()

    if model_name == "linear_svm":
        class_weight = None if svm_class_weight in (None, "none") else svm_class_weight
        return LinearSVC(
            C=svm_c,
            class_weight=class_weight,
            random_state=random_state,
            dual=False,
            max_iter=3000,
        )

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
    use_char_ngrams: bool,
    max_char_features: int,
    numeric_columns: list[str],
    random_state: int,
    svm_c: float = 1.0,
    svm_class_weight: str | None = None,
) -> Pipeline:
    """
    Constroi Pipeline(preprocessor, classifier).

    - word_vectorizer: bag-of-words com bigramas, stopwords PT, limite max_features.
    - char_vectorizer (opcional): n-grams de caracteres com bordas de palavra (char_wb).
    - numeric: StandardScaler sem centralizar (esparsidade/amplitude).
    """
    transformers = [
        (
            "word_vectorizer",
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
    ]

    if use_char_ngrams:
        transformers.append(
            (
                "char_vectorizer",
                CountVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=max_char_features,
                    lowercase=True,
                ),
                "combined_text",
            )
        )

    transformers.append(
        (
            "numeric",
            StandardScaler(with_mean=False),
            numeric_columns,
        )
    )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    estimator = build_estimator(
        model_name=model_name,
        random_state=random_state,
        svm_c=svm_c,
        svm_class_weight=svm_class_weight,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
