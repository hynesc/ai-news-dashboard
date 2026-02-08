from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

if TYPE_CHECKING:
    import pandas as pd


DEFAULT_DOMAIN_STOPWORDS = {
    # Too generic for "AI governance" tracking; tends to dominate phrases.
    "ai",
    "agi",
    "llm",
    "llms",
    "model",
    "models",
    "openai",
    "chatgpt",
    "artificial",
    "intelligence",
    # Headline boilerplate
    "says",
    "say",
    "report",
    "reports",
    "new",
}


def _empty_concepts_df() -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(columns=["Concept", "Score", "Pos", "Neg"])


def _stop_words(extra_stopwords: Optional[Iterable[str]]) -> list[str]:
    base = set(ENGLISH_STOP_WORDS) | set(DEFAULT_DOMAIN_STOPWORDS)
    return sorted(base | set(extra_stopwords or ()))


def _build_vectorizer(
    *,
    ngram_range: tuple[int, int],
    min_df: int,
    max_df: float,
    stop_words: list[str],
) -> CountVectorizer:
    return CountVectorizer(
        # Count per-document presence, not raw term frequency.
        binary=True,
        # Drop pure numbers / 1-char noise (e.g., years, "x", etc.).
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z-]{1,}\b",
        ngram_range=ngram_range,
        stop_words=stop_words,
        min_df=min_df,
        max_df=max_df,
    )


def distinctive_ngrams_df(
    positive_texts: pd.Series,
    negative_texts: pd.Series,
    *,
    ngram_range: tuple[int, int] = (2, 3),
    min_df: int = 2,
    max_df: float = 0.7,
    extra_stopwords: Optional[Iterable[str]] = None,
    alpha: float = 0.5,
    min_total_hits: int = 3,
    min_bucket_hits: int = 2,
) -> pd.DataFrame:
    """Rank n-grams by how strongly they distinguish positive vs negative texts.

    Score is a smoothed log-odds ratio:
      score > 0: more associated with positive bucket
      score < 0: more associated with negative bucket

    `Pos` and `Neg` are counts of *articles* containing the n-gram (not raw term frequency).
    """
    import numpy as np
    import pandas as pd

    if positive_texts is None or negative_texts is None:
        return _empty_concepts_df()
    if getattr(positive_texts, "empty", False) or getattr(negative_texts, "empty", False):
        return _empty_concepts_df()

    vectorizer = _build_vectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words=_stop_words(extra_stopwords),
    )

    combined = pd.concat([positive_texts, negative_texts], ignore_index=True)
    try:
        vectorizer.fit(combined)
        X_pos = vectorizer.transform(positive_texts)
        X_neg = vectorizer.transform(negative_texts)
    except ValueError:
        return _empty_concepts_df()

    pos = np.asarray(X_pos.sum(axis=0)).ravel().astype(float)
    neg = np.asarray(X_neg.sum(axis=0)).ravel().astype(float)
    vocab = vectorizer.get_feature_names_out()

    pos_total = float(pos.sum())
    neg_total = float(neg.sum())
    V = float(len(vocab))
    if pos_total <= 0 or neg_total <= 0 or V <= 0:
        return _empty_concepts_df()

    denom_pos = pos_total + alpha * V
    denom_neg = neg_total + alpha * V
    score = np.log((pos + alpha) / denom_pos) - np.log((neg + alpha) / denom_neg)

    out = pd.DataFrame(
        {
            "Concept": vocab,
            "Score": score,
            "Pos": pos.astype(int),
            "Neg": neg.astype(int),
        }
    )

    total = out["Pos"] + out["Neg"]
    out = out[
        (total >= int(min_total_hits)) & (out[["Pos", "Neg"]].max(axis=1) >= int(min_bucket_hits))
    ]
    return out.sort_values("Score", ascending=False, kind="mergesort").reset_index(drop=True)
