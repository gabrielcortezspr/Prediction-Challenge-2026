"""
Engenharia de features a partir de combined_text, title, text e colunas de ASIN.

Produz um DataFrame com:
- combined_text (passa direto ao vetorizador no pipeline);
- dezenas de colunas numericas (sentimento lexico, flags de tema, comprimentos, etc.).

get_numeric_feature_columns escolhe o subconjunto usado pelo StandardScaler no model.py.
"""
from __future__ import annotations

import re

import pandas as pd

# Lexicos simples de polaridade para contagem e score de sentimento.
POSITIVE_WORDS = {
    "bom",
    "boa",
    "bons",
    "boas",
    "otimo",
    "otima",
    "ótimo",
    "ótima",
    "excelente",
    "recomendo",
    "gostei",
    "adorei",
    "perfeito",
    "perfeita",
    "maravilhoso",
    "maravilhosa",
    "amei",
    "satisfeito",
    "satisfeita",
    "qualidade",
    "beneficio",
    "benefício",
    "aprovado",
    "funciona",
    "funcionou",
    "duravel",
    "durável",
}

NEGATIVE_WORDS = {
    "ruim",
    "péssimo",
    "pessimo",
    "péssima",
    "pessima",
    "horrível",
    "horrivel",
    "decepcao",
    "decepção",
    "decepcionante",
    "nao",
    "não",
    "nunca",
    "absurdo",
    "rasgado",
    "quebrou",
    "defeito",
    "defeituoso",
    "fraco",
    "travando",
    "frustrante",
    "caro",
    "lento",
    "atraso",
    "demorou",
}

# Regex auxiliares: palavras Unicode, tokens alfabeticos, palavras em maiusculas.
WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)
TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÿ]+", flags=re.UNICODE)
CAPS_PATTERN = re.compile(r"\b[A-ZÀ-Ý]{2,}\b", flags=re.UNICODE)

# Conectivos de contraste — usados em antithesis_norm_count e para filtrar stopwords.
ANTITHESIS_TERMS = [
    "mas",
    "porém",
    "porem",
    "entretanto",
    "contudo",
    "todavia",
    "apesar de",
    "não obstante",
    "nao obstante",
]

DEFECT_TERMS = [
    "quebrou",
    "rasgado",
    "defeito",
    "defeituoso",
    "faltando",
    "faltou",
    "nao funciona",
    "não funciona",
    "nao lig",
    "não lig",
    "travando",
    "deteriorado",
]

PRICE_TERMS = [
    "preco",
    "preço",
    "caro",
    "barato",
    "custo",
    "valor",
    "abusivo",
    "custo beneficio",
    "custo-beneficio",
    "custo-benefício",
    "nao vale",
    "não vale",
]

DELIVERY_TERMS = [
    "entrega",
    "entregue",
    "chegou",
    "prazo",
    "rastre",
    "postagem",
    "atras",
    "atraso",
    "demorou",
]

NEGATION_TERMS = {"nao", "não", "nunca", "nem", "jamais"}


def _word_count(text: str) -> int:
    """Numero de tokens alfanumericos (limite de palavra) no texto."""
    return len(WORD_PATTERN.findall(text))


def _char_count(text: str) -> int:
    """Comprimento bruto da string."""
    return len(text)


def _exclamation_count(text: str) -> int:
    return text.count("!")


def _question_count(text: str) -> int:
    return text.count("?")


def _caps_word_count(text: str) -> int:
    """Palavras inteiramente em maiusculas (2+ letras), ex.: GOSTEI."""
    return len(CAPS_PATTERN.findall(text))


def _tokenize_lower(text: str) -> list[str]:
    """Tokens so com letras (incl. acentos), em minusculas — alinha com lexicos."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _count_lexicon_hits(tokens: list[str], lexicon: set[str]) -> int:
    return sum(1 for token in tokens if token in lexicon)


def _count_pattern_occurrences(lowered: str, terms: list[str]) -> int:
    """Conta ocorrencias de termos multi-palavra (substring) ou palavra (regex com borda)."""
    total = 0
    for term in terms:
        if " " in term:
            total += lowered.count(term)
        else:
            total += len(re.findall(rf"\\b{re.escape(term)}", lowered))
    return total


def _sentiment_score(text: str) -> float:
    """Diferenca normalizada (#positivos - #negativos) / num_tokens."""
    tokens = _tokenize_lower(text)
    if not tokens:
        return 0.0

    positive_hits = _count_lexicon_hits(tokens, POSITIVE_WORDS)
    negative_hits = _count_lexicon_hits(tokens, NEGATIVE_WORDS)

    return (positive_hits - negative_hits) / max(len(tokens), 1)


def _antithesis_norm_count(text: str, word_count: int) -> float:
    """Densidade de conectivos de contraste por palavra (review mista)."""
    lowered = text.lower()
    occurrences = _count_pattern_occurrences(lowered, ANTITHESIS_TERMS)
    return occurrences / max(word_count, 1)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constroi todas as colunas esperadas pelo pipeline: texto + numericas derivadas.

    Depende de preprocessing.prepare_datasets (combined_text, asin_encoded, asin_freq).
    """
    features = pd.DataFrame(index=df.index)

    text_series = df["combined_text"].fillna("").astype(str)
    title_series = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
    body_series = df.get("text", pd.Series("", index=df.index)).fillna("").astype(str)
    lowered_series = text_series.str.lower()
    token_series = text_series.apply(_tokenize_lower)

    # Coluna de texto consumida pelo ColumnTransformer (nao e escalada).
    features["combined_text"] = text_series

    # Estatisticas de comprimento e pontuacao.
    features["word_count"] = text_series.apply(_word_count)
    features["char_count"] = text_series.apply(_char_count)
    features["exclamation_count"] = text_series.apply(_exclamation_count)
    features["question_count"] = text_series.apply(_question_count)
    features["caps_word_count"] = text_series.apply(_caps_word_count)

    # Sentimento lexico e indicadores de polaridade mista.
    features["sentiment_score"] = text_series.apply(_sentiment_score)
    features["positive_word_count"] = token_series.apply(
        lambda tokens: _count_lexicon_hits(tokens, POSITIVE_WORDS)
    )
    features["negative_word_count"] = token_series.apply(
        lambda tokens: _count_lexicon_hits(tokens, NEGATIVE_WORDS)
    )
    features["sentiment_ratio"] = (
        (features["positive_word_count"] + 1.0)
        / (features["negative_word_count"] + 1.0)
    )
    features["mixed_sentiment_flag"] = (
        (features["positive_word_count"] > 0)
        & (features["negative_word_count"] > 0)
    ).astype(int)

    # Temas frequentes em reviews de produto: defeito, preco, entrega; negacao.
    features["defect_term_count"] = lowered_series.apply(
        lambda text: _count_pattern_occurrences(text, DEFECT_TERMS)
    )
    features["defect_flag"] = (features["defect_term_count"] > 0).astype(int)
    features["price_term_count"] = lowered_series.apply(
        lambda text: _count_pattern_occurrences(text, PRICE_TERMS)
    )
    features["price_flag"] = (features["price_term_count"] > 0).astype(int)
    features["delivery_term_count"] = lowered_series.apply(
        lambda text: _count_pattern_occurrences(text, DELIVERY_TERMS)
    )
    features["delivery_flag"] = (features["delivery_term_count"] > 0).astype(int)
    features["negation_count"] = token_series.apply(
        lambda tokens: _count_lexicon_hits(tokens, NEGATION_TERMS)
    )

    # Relacao titulo vs corpo e taxas por palavra.
    features["title_word_count"] = title_series.apply(_word_count)
    features["text_word_count"] = body_series.apply(_word_count)
    features["title_to_text_ratio"] = (
        features["title_word_count"] / (features["text_word_count"] + 1.0)
    )
    features["char_per_word"] = (
        features["char_count"] / features["word_count"].clip(lower=1)
    )
    features["exclamation_rate"] = (
        features["exclamation_count"] / features["word_count"].clip(lower=1)
    )
    features["question_rate"] = (
        features["question_count"] / features["word_count"].clip(lower=1)
    )
    features["caps_rate"] = (
        features["caps_word_count"] / features["word_count"].clip(lower=1)
    )

    # Sinais de produto (do preprocessamento).
    features["asin_encoded"] = df["asin_encoded"].astype(int)
    if "asin_freq" in df.columns:
        features["asin_freq"] = df["asin_freq"].astype(float)
    else:
        features["asin_freq"] = 0.0

    # Interacoes / sinais compostos.
    features["sentiment_x_word_count"] = (
        features["sentiment_score"] * features["word_count"]
    )
    features["antithesis_norm_count"] = text_series.combine(
        features["word_count"], _antithesis_norm_count
    )

    return features


def get_numeric_feature_columns(mode: str = "full") -> list[str]:
    """
    Lista de colunas passadas ao StandardScaler (exclui combined_text).

    mode='full': todas as numericas hand-crafted.
    mode='minimal': subconjunto reduzido para menos ruido.
    """
    full_columns = [
        "word_count",
        "char_count",
        "exclamation_count",
        "question_count",
        "caps_word_count",
        "exclamation_rate",
        "question_rate",
        "caps_rate",
        "char_per_word",
        "sentiment_score",
        "positive_word_count",
        "negative_word_count",
        "sentiment_ratio",
        "mixed_sentiment_flag",
        "defect_term_count",
        "defect_flag",
        "price_term_count",
        "price_flag",
        "delivery_term_count",
        "delivery_flag",
        "negation_count",
        "asin_encoded",
        "asin_freq",
        "title_word_count",
        "text_word_count",
        "title_to_text_ratio",
        "sentiment_x_word_count",
        "antithesis_norm_count",
    ]

    if mode == "full":
        return full_columns

    if mode == "minimal":
        # Minimal set to reduce hand-crafted noise and keep robust signals.
        return [
            "word_count",
            "sentiment_score",
            "antithesis_norm_count",
            "asin_freq",
        ]

    raise ValueError("mode deve ser 'full' ou 'minimal'")
