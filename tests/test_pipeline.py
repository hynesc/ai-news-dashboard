from __future__ import annotations

import pandas as pd

from ai_news_dashboard.pipeline import _sentiment_text, dedupe_articles


def test_sentiment_text_prefers_description_and_trims() -> None:
    article = {"title": " Hello ", "description": " World "}
    assert _sentiment_text(article) == "Hello. World"


def test_dedupe_prefers_url() -> None:
    df = pd.DataFrame(
        [
            {"title": "a", "url": "https://example.com/1", "source": "x", "published_at": "2025-01-01"},
            {"title": "a (dupe)", "url": "https://example.com/1", "source": "y", "published_at": "2025-01-02"},
        ]
    )
    out = dedupe_articles(df)
    assert len(out) == 1
    assert out.iloc[0]["url"] == "https://example.com/1"


def test_dedupe_fallback_works_without_url() -> None:
    df = pd.DataFrame(
        [
            {"title": "Same", "source": "S", "published_at": "2025-01-01"},
            {"title": "Same", "source": "S", "published_at": "2025-01-01"},
        ]
    )
    out = dedupe_articles(df)
    assert len(out) == 1


def test_dedupe_handles_missing_columns() -> None:
    df = pd.DataFrame([{"x": 1}, {"x": 1}])
    out = dedupe_articles(df)
    # Without any identifying columns, we can't reliably dedupe; we should not crash.
    assert len(out) == 2

