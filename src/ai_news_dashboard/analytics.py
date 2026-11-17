from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from ai_news_dashboard.text_analytics import distinctive_ngrams_df

TOPIC_PATTERNS: dict[str, str] = {
    "Regulation & Policy": (
        r"\b(?:"
        r"regulation|regulatory|policy|governance|lawmakers?|legislation|executive order"
        r")\b"
    ),
    "EU AI Act": r"\b(?:eu ai act|ai act|european union)\b",
    "Copyright & IP": (
        r"\b(?:copyright|intellectual property|ip|licensing|fair use|training data)\b"
    ),
    "Misinformation & Deepfakes": (
        r"\b(?:misinformation|disinformation|deepfakes?|synthetic media)\b"
    ),
    "Bias & Fairness": r"\b(?:bias|fairness|discrimination|equity)\b",
    "Privacy & Surveillance": (
        r"\b(?:privacy|data protection|gdpr|surveillance|biometric|facial recognition)\b"
    ),
    "Safety & Alignment": (
        r"\b(?:safety|alignment|rlhf|interpretability|evals?|red[- ]teaming|robustness)\b"
    ),
    "Existential Risk": (
        r"\b(?:x[- ]?risk|existential risk|catastroph(?:e|ic)|doomsday|extinction)\b"
    ),
    "Compute & Chips": r"\b(?:chips?|gpus?|compute|semiconductor|nvidia|amd|export controls?)\b",
    "Open Source": r"\b(?:open source|open[- ]weight|weights released|model release)\b",
    "Security & Cyber": r"\b(?:security|cyber|hacking|malware|phishing|vulnerabilit(?:y|ies))\b",
    "Jobs & Labor": r"\b(?:jobs?|labor|workers?|unions?|automation|layoffs?)\b",
    "Big Tech & Antitrust": r"\b(?:antitrust|competition|monopoly|ftc|doj)\b",
    "Litigation": r"\b(?:lawsuit|sued|court|judge|settlement|legal challenge)\b",
    "Weapons & Military": r"\b(?:military|defense|weapons?|autonomous weapons?|pentagon|warfare)\b",
}


def build_concept_text(df: pd.DataFrame) -> pd.Series:
    title = (
        df["title"].fillna("").astype(str)
        if "title" in df.columns
        else pd.Series("", index=df.index)
    )
    snippet = (
        df["content_snippet"].fillna("").astype(str)
        if "content_snippet" in df.columns
        else pd.Series("", index=df.index)
    )

    text = (title + ". " + snippet).str.strip()
    # NewsAPI content often ends with "… [+123 chars]".
    text = text.str.replace(r"\s*\[\+\d+\schars\]\s*$", "", regex=True)
    text = text.str.replace(r"https?://\S+", " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True)
    return text


def build_all_terms_regex(terms: list[str]) -> str:
    # Positive lookaheads let us do an AND search for all terms in any order.
    return "".join([rf"(?=.*\b{re.escape(t)}\b)" for t in terms if t.strip()])


def cap_articles_per_source(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if cap <= 0 or "source" not in df.columns:
        return df
    sort_col = "published_at" if "published_at" in df.columns else None
    if sort_col:
        return (
            df.sort_values(sort_col, na_position="first").groupby("source", dropna=False).tail(cap)
        )
    return df.groupby("source", dropna=False).tail(cap)


def topic_scores_df(pos_text: pd.Series, neg_text: pd.Series) -> pd.DataFrame:
    import numpy as np

    rows = []
    for topic, pattern in TOPIC_PATTERNS.items():
        pos = int(pos_text.str.contains(pattern, case=False, na=False).sum())
        neg = int(neg_text.str.contains(pattern, case=False, na=False).sum())
        if pos + neg == 0:
            continue
        score = np.log((pos + 0.5) / (len(pos_text) - pos + 0.5)) - np.log(
            (neg + 0.5) / (len(neg_text) - neg + 0.5)
        )
        rows.append({"Concept": topic, "Score": score, "Pos": pos, "Neg": neg})
    return (
        pd.DataFrame(rows, columns=["Concept", "Score", "Pos", "Neg"])
        .sort_values("Score", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )


def compute_ranked_concepts(
    *,
    concept_mode: str,
    positive_articles: pd.DataFrame,
    negative_articles: pd.DataFrame,
    ngram_min_df: int,
    ngram_min_hits: int,
    ngram_min_total: int,
) -> pd.DataFrame:
    pos_text = build_concept_text(positive_articles)
    neg_text = build_concept_text(negative_articles)

    if concept_mode == "Topics (general)":
        return topic_scores_df(pos_text, neg_text)

    return distinctive_ngrams_df(
        pos_text,
        neg_text,
        ngram_range=(2, 3),
        min_df=int(ngram_min_df),
        max_df=0.7,
        min_bucket_hits=int(ngram_min_hits),
        min_total_hits=int(ngram_min_total),
    )


def concept_to_pattern(concept_mode: str, selected_concept: str) -> str:
    if concept_mode == "Topics (general)":
        return TOPIC_PATTERNS.get(selected_concept, re.escape(selected_concept))
    return build_all_terms_regex(selected_concept.split())


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep malformed rows from breaking the UI; retain articles without scores or dates."""
    df = df.copy()
    for column in ["title", "source", "content_snippet", "url"]:
        if column not in df:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str).str.strip()
    df = df[df["title"].ne("") & df["title"].ne("[Removed]")].copy()
    df["source"] = df["source"].replace("", "Unknown source")
    # Parse each value independently, including mixed timezone offsets and formats.
    df["published_at"] = pd.to_datetime(
        df.get("published_at", pd.Series(index=df.index, dtype=str)),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    df["sentiment_compound"] = pd.to_numeric(
        df.get("sentiment_compound", pd.Series(index=df.index, dtype=float)), errors="coerce"
    )
    df.loc[~df["sentiment_compound"].between(-1, 1), "sentiment_compound"] = float("nan")
    df.loc[~df["url"].str.match(r"https?://", case=False), "url"] = None
    return df.sort_values("published_at", ascending=False, na_position="last").reset_index(
        drop=True
    )


def filter_by_timeframe(df: pd.DataFrame, days: Optional[int]) -> pd.DataFrame:
    if days is None:
        return df
    now = pd.Timestamp.now(tz="UTC")
    return df[df["published_at"].between(now - pd.Timedelta(days=days), now)]


def split_sentiment_buckets(df: pd.DataFrame, q: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    sent = df["sentiment_compound"].dropna()
    lo, hi = sent.quantile(q), sent.quantile(1 - q)
    # Neutral articles and ties must never appear on both sides of a comparison.
    return (
        df[(df["sentiment_compound"] >= hi) & (df["sentiment_compound"] > 0.05)],
        df[(df["sentiment_compound"] <= lo) & (df["sentiment_compound"] < -0.05)],
    )


def sentiment_trend(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.dropna(subset=["published_at"])
        .set_index("published_at")
        .resample("D")["sentiment_compound"]
        .mean()
        .rename("Daily mean")
        .to_frame()
    )
    daily["7-day moving average"] = daily["Daily mean"].rolling("7D", min_periods=1).mean()
    return daily.reset_index()
