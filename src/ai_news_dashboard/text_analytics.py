from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.feature_extraction.text import CountVectorizer

if TYPE_CHECKING:
    import pandas as pd


def top_ngrams_df(
    texts: "pd.Series",
    *,
    ngram_range: tuple[int, int] = (2, 3),
    n_top: int = 10,
) -> "pd.DataFrame":
    """Compute the most frequent n-grams in `texts` (typically headlines)."""
    import pandas as pd

    if texts is None or getattr(texts, "empty", False):
        return pd.DataFrame(columns=["Concept", "Frequency"])

    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english")
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=["Concept", "Frequency"])

    sum_words = X.sum(axis=0)
    words_freq = [(word, int(sum_words[0, idx])) for word, idx in vectorizer.vocabulary_.items()]
    words_freq.sort(key=lambda x: x[1], reverse=True)
    return pd.DataFrame(words_freq[:n_top], columns=["Concept", "Frequency"])

