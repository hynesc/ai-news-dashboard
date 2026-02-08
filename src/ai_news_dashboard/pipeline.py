from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from ai_news_dashboard.config import Settings
from ai_news_dashboard.logging_utils import setup_logging
from ai_news_dashboard.nltk_resources import ensure_nltk_data

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArticleRow:
    source: Optional[str]
    title: str
    url: Optional[str]
    published_at: Optional[datetime]
    content_snippet: str
    sentiment_compound: float


def _safe_parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return pd.to_datetime(value, utc=True).to_pydatetime()
    except Exception:
        return None


def _sentiment_text(article: dict[str, Any]) -> str:
    title = (article.get("title") or "").strip()
    description = (article.get("description") or article.get("content") or "").strip()
    if description:
        return f"{title}. {description}".strip()
    return title


def fetch_articles(
    api: NewsApiClient,
    *,
    query: str,
    language: str,
    sort_by: str,
    page_size: int,
    max_pages: int,
) -> list[dict[str, Any]]:
    articles: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        log.info("Fetching page %s for query (%s chars)", page, len(query))
        try:
            resp = api.get_everything(
                q=query,
                language=language,
                sort_by=sort_by,
                page_size=page_size,
                page=page,
            )
        except Exception:
            log.exception("NewsAPI request failed on page %s", page)
            break

        batch = resp.get("articles") or []
        articles.extend(batch)
        if len(batch) < page_size:
            break
    return articles


def analyze_sentiment(articles: list[dict[str, Any]]) -> pd.DataFrame:
    ensure_nltk_data(["vader_lexicon"])
    sid = SentimentIntensityAnalyzer()

    rows: list[dict[str, Any]] = []
    for article in articles:
        text = _sentiment_text(article)
        if not text.strip():
            continue

        sentiment = sid.polarity_scores(text)["compound"]
        rows.append(
            ArticleRow(
                source=(article.get("source") or {}).get("name"),
                title=(article.get("title") or "").strip(),
                url=article.get("url"),
                published_at=_safe_parse_datetime(article.get("publishedAt")),
                content_snippet=(
                    (article.get("description") or article.get("content") or "").strip()
                ),
                sentiment_compound=float(sentiment),
            ).__dict__
        )

    df = pd.DataFrame(rows)
    if not df.empty and "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


def dedupe_articles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if not {"url", "title", "source", "published_at"} & set(df.columns):
        # No stable keys available.
        return df

    df = df.copy()
    idx = df.index
    # Prefer URL for dedupe when available; fall back to title+source+published_at.
    has_url = df.get("url").notna() if "url" in df.columns else pd.Series(False, index=df.index)
    df["__dedupe_key"] = None
    df.loc[has_url, "__dedupe_key"] = df.loc[has_url, "url"].astype(str).str.strip().str.lower()

    title = (
        df["title"].fillna("").astype(str).str.strip().str.lower()
        if "title" in df.columns
        else pd.Series("", index=idx)
    )
    source = (
        df["source"].fillna("").astype(str).str.strip().str.lower()
        if "source" in df.columns
        else pd.Series("", index=idx)
    )
    published = (
        pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        .fillna("")
        if "published_at" in df.columns
        else pd.Series("", index=idx)
    )

    fallback = (title + "||" + source + "||" + published).where(df["__dedupe_key"].isna())
    df["__dedupe_key"] = df["__dedupe_key"].fillna(fallback)

    df = df.drop_duplicates(subset=["__dedupe_key"], keep="last").drop(columns=["__dedupe_key"])
    return df


def load_existing(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=["published_at"])
        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def run(settings: Settings) -> int:
    if not settings.newsapi_key:
        log.error("NEWSAPI_KEY environment variable not set.")
        return 2

    api = NewsApiClient(api_key=settings.newsapi_key)
    existing = load_existing(str(settings.output_csv))

    articles = fetch_articles(
        api,
        query=settings.search_query,
        language=settings.language,
        sort_by=settings.sort_by,
        page_size=settings.page_size,
        max_pages=settings.max_pages,
    )
    if not articles:
        log.info("No new articles fetched.")
        return 0

    new = analyze_sentiment(articles)
    combined = pd.concat([existing, new], ignore_index=True)
    combined = dedupe_articles(combined)
    save_csv(combined, str(settings.output_csv))
    log.info("Saved %s total articles to %s", len(combined), settings.output_csv)
    return 0


def main() -> int:
    setup_logging()
    return run(Settings.from_env())
