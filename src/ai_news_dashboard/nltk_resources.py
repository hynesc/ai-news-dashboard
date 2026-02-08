from __future__ import annotations

from collections.abc import Iterable
import logging

import nltk

log = logging.getLogger(__name__)


_RESOURCE_PATHS: dict[str, list[str]] = {
    # Tokenizers
    "punkt": ["tokenizers/punkt", "tokenizers/punkt.zip"],
    # Corpora
    "stopwords": ["corpora/stopwords", "corpora/stopwords.zip"],
    "wordnet": ["corpora/wordnet", "corpora/wordnet.zip"],
    # Sentiment lexicon
    "vader_lexicon": ["sentiment/vader_lexicon", "sentiment/vader_lexicon.zip"],
}


def ensure_nltk_data(resources: Iterable[str]) -> None:
    """Ensure required NLTK datasets exist, downloading if missing.

    This is safe to call multiple times.
    """
    for resource in resources:
        paths = _RESOURCE_PATHS.get(resource)
        if not paths:
            raise ValueError(f"Unknown NLTK resource: {resource!r}")

        for path in paths:
            try:
                nltk.data.find(path)
                break
            except LookupError:
                continue
        else:
            log.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)
