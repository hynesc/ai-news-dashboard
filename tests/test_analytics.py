from __future__ import annotations

import numpy as np
import pandas as pd

from ai_news_dashboard.analytics import (
    build_concept_text,
    normalize_data,
    sentiment_trend,
    split_sentiment_buckets,
    topic_scores_df,
)
from ai_news_dashboard.text_analytics import distinctive_ngrams_df


def test_normalize_mixed_dates_and_bad_values():
    df = normalize_data(
        pd.DataFrame(
            {
                "title": ["a", "b", "[Removed]", "c"],
                "published_at": ["2025-01-01", "2025-01-02T02:00:00+02:00", "bad", "bad"],
                "sentiment_compound": ["0.2", "bad", 0, 20],
                "url": ["https://example.com", "javascript:alert(1)", "", ""],
            }
        )
    )
    assert len(df) == 3
    assert df["published_at"].notna().sum() == 2
    assert df["sentiment_compound"].notna().sum() == 1
    assert df["url"].notna().sum() == 1
    assert df["source"].eq("Unknown source").all()


def test_buckets_exclude_neutral_and_never_overlap():
    df = pd.DataFrame({"sentiment_compound": [-0.8, 0, 0, 0, 0, 0.8]})
    pos, neg = split_sentiment_buckets(df, 0.4)
    assert len(pos) == len(neg) == 1
    assert pos.index.intersection(neg.index).empty
    neutral = pd.DataFrame({"sentiment_compound": [0, 0, 0]})
    assert all(bucket.empty for bucket in split_sentiment_buckets(neutral, 0.2))


def test_trend_uses_calendar_days():
    df = normalize_data(
        pd.DataFrame(
            {
                "title": ["a", "b"],
                "published_at": ["2025-01-01", "2025-01-10"],
                "sentiment_compound": [1, -1],
            }
        )
    )
    trend = sentiment_trend(df)
    assert trend.iloc[-1]["7-day moving average"] == -1
    assert len(trend) == 10


def test_topic_scores_normalize_bucket_sizes_and_drop_absent_topics():
    ranked = topic_scores_df(pd.Series(["safety", "other"] * 5), pd.Series(["safety", "other"]))
    assert ranked["Concept"].tolist() == ["Safety & Alignment"]
    assert np.isclose(ranked.iloc[0]["Score"], 0)


def test_phrase_scores_normalize_document_counts():
    ranked = distinctive_ngrams_df(
        pd.Series(["safety testing"] * 5 + ["policy review"] * 5),
        pd.Series(["safety testing", "policy review"]),
        min_df=1,
        max_df=1.0,
        min_bucket_hits=1,
        min_total_hits=1,
    )
    assert not ranked.empty
    assert np.allclose(ranked["Score"], 0)


def test_concept_text_handles_missing_columns():
    result = build_concept_text(pd.DataFrame(index=[0, 1]))
    assert len(result) == 2
