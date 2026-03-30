from __future__ import annotations

import re
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder


BASE_REQUIRED_COLUMNS = ["id", "ASIN", "title", "text"]
TEST_REQUIRED_COLUMNS = ["id", "ASIN", "title", "text"]

PORTUGUESE_STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos",
    "em", "no", "na", "nos", "nas",
    "para", "por", "com", "sem",
    "e", "ou", "que", "como", "muito", "muita", "muitos", "muitas",
    "se", "isso", "essa", "esse", "esses", "essas", "isto", "aquilo",
    "foi", "ser", "ter", "sou", "era", "sao", "são", "é",
    "ao", "aos", "à", "às", "seu", "sua", "seus", "suas",
    "meu", "minha", "meus", "minhas", "você", "vocês", "eu", "nós",
}


def normalize_text_for_vectorizer(text: str) -> str:
    """Normaliza texto para vetorizacao sem remover sinal linguistico relevante."""
    normalized = str(text).lower()
    normalized = re.sub(r"http\S+|www\S+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def get_portuguese_stopwords() -> list[str]:
    """Retorna stopwords em portugues preservando conectivos de antitese."""
    from features import ANTITHESIS_TERMS

    antithesis_single_terms = {
        term.lower()
        for term in ANTITHESIS_TERMS
        if len(term.split()) == 1
    }
    filtered_stopwords = PORTUGUESE_STOPWORDS - antithesis_single_terms
    return sorted(filtered_stopwords)


def _validate_columns(df: pd.DataFrame, required_columns: list[str], file_label: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{file_label} sem colunas obrigatorias: {missing}")


def load_train_test(
    train_path: str,
    test_path: str,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_required_columns = BASE_REQUIRED_COLUMNS + [target_column]
    _validate_columns(train_df, train_required_columns, "train file")
    _validate_columns(test_df, TEST_REQUIRED_COLUMNS, "test.csv")

    return train_df, test_df


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned["title"] = cleaned["title"].fillna("").astype(str).str.strip()
    cleaned["text"] = cleaned["text"].fillna("").astype(str).str.strip()
    cleaned["ASIN"] = cleaned["ASIN"].fillna("UNKNOWN").astype(str).str.strip()

    # No dataset, alguns titulos aparecem como '.' representando ruido.
    cleaned["title"] = cleaned["title"].replace(".", "", regex=False)

    cleaned["combined_text"] = (cleaned["title"] + " " + cleaned["text"]).str.strip()
    cleaned["combined_text"] = cleaned["combined_text"].replace("", "sem comentario")

    return cleaned


def fit_asin_encoder(train_df: pd.DataFrame) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(train_df["ASIN"].astype(str))
    return encoder


def transform_asin(df: pd.DataFrame, encoder: LabelEncoder) -> pd.Series:
    asin_to_int = {asin: idx for idx, asin in enumerate(encoder.classes_)}
    return df["ASIN"].map(asin_to_int).fillna(-1).astype(int)


def fit_asin_frequency(train_df: pd.DataFrame) -> pd.Series:
    return train_df["ASIN"].value_counts(normalize=True)


def transform_asin_frequency(df: pd.DataFrame, asin_frequency: pd.Series) -> pd.Series:
    return df["ASIN"].map(asin_frequency).fillna(0.0).astype(float)


def prepare_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    train_clean = clean_text_fields(train_df)
    test_clean = clean_text_fields(test_df)

    asin_encoder = fit_asin_encoder(train_clean)
    asin_frequency = fit_asin_frequency(train_clean)

    train_clean["asin_encoded"] = transform_asin(train_clean, asin_encoder)
    test_clean["asin_encoded"] = transform_asin(test_clean, asin_encoder)
    train_clean["asin_freq"] = transform_asin_frequency(train_clean, asin_frequency)
    test_clean["asin_freq"] = transform_asin_frequency(test_clean, asin_frequency)

    return train_clean, test_clean, asin_encoder
