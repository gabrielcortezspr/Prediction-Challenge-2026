from __future__ import annotations

import re

import pandas as pd


POSITIVE_WORDS = {
    "bom",
    "boa",
    "otimo",
    "otima",
    "excelente",
    "recomendo",
    "gostei",
    "perfeito",
    "maravilhoso",
    "amei",
    "satisfeito",
    "qualidade",
}

NEGATIVE_WORDS = {
    "ruim",
    "pessimo",
    "pessima",
    "horrivel",
    "decepcao",
    "decepcionante",
    "nao",
    "nunca",
    "absurdo",
    "rasgado",
    "quebrou",
    "travando",
    "frustrante",
}

WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)
TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÿ]+", flags=re.UNICODE)
CAPS_PATTERN = re.compile(r"\b[A-ZÀ-Ý]{2,}\b", flags=re.UNICODE)


def _word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def _char_count(text: str) -> int:
    return len(text)


def _exclamation_count(text: str) -> int:
    return text.count("!")


def _question_count(text: str) -> int:
    return text.count("?")


def _caps_word_count(text: str) -> int:
    return len(CAPS_PATTERN.findall(text))


def _sentiment_score(text: str) -> float:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    if not tokens:
        return 0.0

    positive_hits = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative_hits = sum(1 for token in tokens if token in NEGATIVE_WORDS)

    return (positive_hits - negative_hits) / max(len(tokens), 1)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    text_series = df["combined_text"].fillna("").astype(str)

    features["combined_text"] = text_series
    features["word_count"] = text_series.apply(_word_count)
    features["char_count"] = text_series.apply(_char_count)
    features["exclamation_count"] = text_series.apply(_exclamation_count)
    features["question_count"] = text_series.apply(_question_count)
    features["caps_word_count"] = text_series.apply(_caps_word_count)
    features["sentiment_score"] = text_series.apply(_sentiment_score)
    features["asin_encoded"] = df["asin_encoded"].astype(int)
    features["sentiment_x_word_count"] = (
        features["sentiment_score"] * features["word_count"]
    )

    return features


def get_numeric_feature_columns() -> list[str]:
    return [
        "word_count",
        "char_count",
        "exclamation_count",
        "question_count",
        "caps_word_count",
        "sentiment_score",
        "asin_encoded",
        "sentiment_x_word_count",
    ]
