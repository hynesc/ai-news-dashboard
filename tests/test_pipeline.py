from __future__ import annotations

import pandas as pd

from ai_news_dashboard.pipeline import _sentiment_text, dedupe_articles


def test_sentiment_text_prefers_description_and_trims() -> None:
    article = {"title": " Hello ", "description": " World "}
    assert _sentiment_text(article) == "Hello. World"


def test_dedupe_prefers_url() -> None:
    df = pd.DataFrame(
        [
            {
                "title": "a",
                "url": "https://example.com/1",
                "source": "x",
                "published_at": "2025-01-01",
            },
            {
                "title": "a (dupe)",
                "url": "https://example.com/1",
                "source": "y",
                "published_at": "2025-01-02",
            },
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


def test_blank_urls_use_fallback_and_url_paths_preserve_case():
    df = pd.DataFrame(
        [
            {"title": "a", "url": " "},
            {"title": "b", "url": ""},
            {"title": "c", "url": "https://example.com/Path"},
            {"title": "d", "url": "https://example.com/path"},
        ]
    )
    assert len(dedupe_articles(df)) == 4


def test_fetch_failures_are_not_silent():
    import pytest

    from ai_news_dashboard.pipeline import fetch_articles

    class BrokenAPI:
        def get_everything(self, **kwargs):
            raise RuntimeError("API unavailable")

    with pytest.raises(RuntimeError, match="API unavailable"):
        fetch_articles(
            BrokenAPI(),
            query="test",
            language="en",
            sort_by="relevancy",
            page_size=100,
            max_pages=1,
        )


def test_save_is_atomic_on_failure(tmp_path, monkeypatch):
    import pytest

    from ai_news_dashboard.pipeline import save_csv

    path = tmp_path / "news.csv"
    path.write_text("original")

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError):
        save_csv(pd.DataFrame({"title": ["test"]}), str(path))
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]
