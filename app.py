from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud


def _ensure_src_on_path() -> None:
    src_dir = Path(__file__).resolve().parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_src_on_path()

from ai_news_dashboard.nltk_resources import ensure_nltk_data  # noqa: E402
from ai_news_dashboard.text_analytics import distinctive_ngrams_df  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent / "news_data.csv"

TIMEFRAMES: dict[str, Optional[int]] = {
    "Last 7 Days": 7,
    "Last 30 Days": 30,
    "Last 90 Days": 90,
    "All Time": None,
}

TOPIC_PATTERNS: dict[str, str] = {
    "Regulation & Policy": (
        r"\b("
        r"regulation|regulatory|policy|governance|lawmakers?|legislation|executive order"
        r")\b"
    ),
    "EU AI Act": r"\b(eu ai act|ai act|european union)\b",
    "Copyright & IP": r"\b(copyright|intellectual property|ip|licensing|fair use|training data)\b",
    "Misinformation & Deepfakes": r"\b(misinformation|disinformation|deepfakes?|synthetic media)\b",
    "Bias & Fairness": r"\b(bias|fairness|discrimination|equity)\b",
    "Privacy & Surveillance": (
        r"\b(privacy|data protection|gdpr|surveillance|biometric|facial recognition)\b"
    ),
    "Safety & Alignment": (
        r"\b(safety|alignment|rlhf|interpretability|evals?|red[- ]teaming|robustness)\b"
    ),
    "Existential Risk": r"\b(x[- ]?risk|existential risk|catastroph(?:e|ic)|doomsday|extinction)\b",
    "Compute & Chips": r"\b(chips?|gpus?|compute|semiconductor|nvidia|amd|export controls?)\b",
    "Open Source": r"\b(open source|open[- ]weight|weights released|model release)\b",
    "Security & Cyber": r"\b(security|cyber|hacking|malware|phishing|vulnerabilit(?:y|ies))\b",
    "Jobs & Labor": r"\b(jobs?|labor|workers?|unions?|automation|layoffs?)\b",
    "Big Tech & Antitrust": r"\b(antitrust|competition|monopoly|ftc|doj)\b",
    "Litigation": r"\b(lawsuit|sued|court|judge|settlement|legal challenge)\b",
    "Weapons & Military": r"\b(military|defense|weapons?|autonomous weapons?|pentagon|warfare)\b",
}


@st.cache_resource
def _download_nltk_resources() -> None:
    ensure_nltk_data(["stopwords"])


@st.cache_data(ttl=600)
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(
            f"Error: The data file '{path}' was not found. Please run the data collection script."
        )
        return pd.DataFrame()

    if "published_at" in df.columns:
        published = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        # If parsing failed badly, try a more permissive parse (handles mixed formats better).
        if published.isna().mean() > 0.8:
            published = pd.to_datetime(df["published_at"], errors="coerce")
        # Normalize to naive timestamps for consistent filtering.
        if getattr(published.dt, "tz", None) is not None:
            df["published_at"] = published.dt.tz_convert(None)
        else:
            df["published_at"] = published

    return df


def build_concept_text(df: pd.DataFrame) -> pd.Series:
    title = df["title"].fillna("").astype(str) if "title" in df.columns else ""
    snippet = (
        df["content_snippet"].fillna("").astype(str) if "content_snippet" in df.columns else ""
    )

    text = (title + ". " + snippet).str.strip()
    # NewsAPI content often ends with "… [+123 chars]".
    text = text.str.replace(r"\s*\[\+\d+\schars\]\s*$", "", regex=True)
    text = text.str.replace(r"https?://\S+", " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True)
    return text


def build_all_terms_regex(terms: list[str]) -> str:
    # Positive lookaheads let us do an AND search for all terms in any order.
    return "".join([f"(?=.*{re.escape(t)})" for t in terms if t.strip()])


def create_wordcloud_figure(headlines: pd.Series):
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))
    stop_words.update(["ai"])

    text = " ".join(headlines.dropna().astype(str))
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stop_words,
    ).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def filter_by_timeframe(df: pd.DataFrame, days: Optional[int]) -> pd.DataFrame:
    if days is None:
        return df
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=int(days))
    return df[df["published_at"] >= cutoff]


def split_sentiment_buckets(df: pd.DataFrame, q: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    sent = df["sentiment_compound"].dropna()
    lo = float(sent.quantile(q))
    hi = float(sent.quantile(1.0 - q))
    neg = df[df["sentiment_compound"] <= lo]
    pos = df[df["sentiment_compound"] >= hi]
    return pos, neg


def cap_articles_per_source(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if cap <= 0 or "source" not in df.columns:
        return df
    sort_col = "published_at" if "published_at" in df.columns else None
    if sort_col:
        return df.sort_values(sort_col).groupby("source", dropna=False).tail(cap)
    return df.groupby("source", dropna=False).tail(cap)


def topic_scores_df(pos_text: pd.Series, neg_text: pd.Series) -> pd.DataFrame:
    rows = []
    for topic, pattern in TOPIC_PATTERNS.items():
        pos = int(pos_text.str.contains(pattern, case=False, regex=True, na=False).sum())
        neg = int(neg_text.str.contains(pattern, case=False, regex=True, na=False).sum())
        # Ratio > 1 means more common in positive bucket; < 1 means more common in negative.
        score = float((pos + 0.5) / (neg + 0.5))
        rows.append({"Concept": topic, "Score": score, "Pos": pos, "Neg": neg})
    return (
        pd.DataFrame(rows)
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

    def try_rank(
        *,
        ngram_range: tuple[int, int],
        min_df_override: Optional[int] = None,
        max_df_override: Optional[float] = None,
    ) -> pd.DataFrame:
        return distinctive_ngrams_df(
            pos_text,
            neg_text,
            ngram_range=ngram_range,
            min_df=int(min_df_override if min_df_override is not None else ngram_min_df),
            max_df=float(max_df_override if max_df_override is not None else 0.7),
            min_bucket_hits=int(ngram_min_hits),
            min_total_hits=int(ngram_min_total),
        )

    ranked = try_rank(ngram_range=(2, 3))
    if ranked.empty:
        ranked = try_rank(ngram_range=(2, 3), min_df_override=1, max_df_override=1.0)
    if ranked.empty:
        ranked = try_rank(ngram_range=(1, 2), min_df_override=1, max_df_override=1.0)
    return ranked


def concept_to_pattern(concept_mode: str, selected_concept: str) -> str:
    if concept_mode == "Topics (general)":
        return TOPIC_PATTERNS.get(selected_concept, re.escape(selected_concept))
    return build_all_terms_regex(selected_concept.split())


def render_app(df: pd.DataFrame) -> None:
    st.set_page_config(page_title="AI Safety & Governance Tracker", page_icon="🤖", layout="wide")
    st.title("🤖 AI Safety & Governance Tracker")
    st.markdown(
        "An automated dashboard tracking global discourse on AI safety, alignment, and policy."
    )

    if df.empty:
        st.warning("No data to display. The data file is missing or empty.")
        return

    st.sidebar.header("Filters")
    timeframe_options = list(TIMEFRAMES.keys())
    timeframe_key = st.sidebar.selectbox(
        "Select Timeframe",
        options=timeframe_options,
        index=timeframe_options.index("All Time"),
    )

    with st.sidebar.expander("Analysis Settings"):
        bucket_quantile = st.slider("Sentiment bucket size (quantile)", 0.10, 0.40, 0.20, 0.05)
        concept_mode = st.selectbox("Concept method", ["Topics (general)", "Phrases (ngrams)"])
        per_source_cap = st.number_input(
            "Max articles per source (per bucket)",
            min_value=0,
            max_value=50,
            value=10,
            help="Reduces single-source domination. Set to 0 to disable.",
        )
        ngram_min_df = st.number_input("Min phrase frequency", min_value=1, max_value=20, value=2)
        ngram_min_hits = st.number_input(
            "Min phrase hits (bucket)",
            min_value=1,
            max_value=20,
            value=2,
        )
        ngram_min_total = st.number_input(
            "Min phrase hits (total)",
            min_value=1,
            max_value=50,
            value=3,
        )

    st.sidebar.info(
        "\n".join(
            [
                "**Sentiment Score Legend (VADER)**",
                "- **Positive:** > 0.05",
                "- **Neutral:** -0.05 to 0.05",
                "- **Negative:** < -0.05",
            ]
        )
    )

    days = TIMEFRAMES[timeframe_key]
    filtered = filter_by_timeframe(df, days)
    if filtered.empty:
        st.warning(f"No articles found for the selected timeframe: **{timeframe_key}**.")
        return

    st.header(f"Analytics for: {timeframe_key}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Articles", f"{len(filtered):,}")

    avg_sent = filtered["sentiment_compound"].mean()
    if pd.isna(avg_sent):
        col2.metric("Avg. Sentiment Score", "N/A")
    else:
        label = "😐"
        if avg_sent > 0.05:
            label = "😊"
        elif avg_sent < -0.05:
            label = "😟"
        col2.metric("Avg. Sentiment Score", f"{avg_sent:.3f} {label}")

    col3.metric("Unique News Sources", f"{filtered['source'].nunique():,}")

    st.header("Visual Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Sentiment Trend", "Sentiment Distribution", "Sentiment by Source", "Word Cloud"]
    )

    with tab1:
        st.subheader("Sentiment Over Time")
        series = (
            filtered.set_index("published_at").resample("D")["sentiment_compound"].mean().dropna()
        )
        trend = series.rename("daily_mean").to_frame()
        trend["ma_7d"] = trend["daily_mean"].rolling(window=7, min_periods=1).mean()
        trend = trend.reset_index()

        fig = px.line(
            trend,
            x="published_at",
            y=["daily_mean", "ma_7d"],
            labels={"value": "Avg. Sentiment", "published_at": "Date", "variable": "Series"},
            template="plotly_white",
        )
        fig.update_traces(
            selector={"name": "daily_mean"},
            name="Daily mean",
            opacity=0.35,
        )
        fig.update_traces(
            selector={"name": "ma_7d"},
            name="7-day moving average",
            line={"width": 3},
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Distribution of Sentiment Scores")
        fig = px.histogram(filtered, x="sentiment_compound", nbins=50)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Average Sentiment by Top Sources")
        by_source = (
            filtered.groupby("source")["sentiment_compound"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_sentiment", "count": "article_count"})
            .nlargest(15, "article_count")
        )
        fig = px.bar(by_source, y="avg_sentiment", title="Top 15 sources (by article volume)")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Headline Word Cloud")
        st.pyplot(create_wordcloud_figure(filtered["title"]))

    st.header("Key Concept Analysis")
    if filtered["sentiment_compound"].dropna().empty:
        st.info("Sentiment data is missing for the selected timeframe.")
        return

    pos, neg = split_sentiment_buckets(filtered, float(bucket_quantile))
    pos = cap_articles_per_source(pos, int(per_source_cap))
    neg = cap_articles_per_source(neg, int(per_source_cap))

    ranked = compute_ranked_concepts(
        concept_mode=concept_mode,
        positive_articles=pos,
        negative_articles=neg,
        ngram_min_df=int(ngram_min_df),
        ngram_min_hits=int(ngram_min_hits),
        ngram_min_total=int(ngram_min_total),
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Concepts Most Associated With Positive Sentiment")
        if ranked.empty:
            st.info(
                "Not enough data to generate concepts. Try a longer timeframe or relax thresholds."
            )
            st.dataframe(
                pd.DataFrame(columns=["Concept", "Score", "Pos", "Neg"]),
                use_container_width=True,
            )
        else:
            st.dataframe(
                ranked.head(15)[["Concept", "Score", "Pos", "Neg"]],
                use_container_width=True,
            )
    with col2:
        st.subheader("Concepts Most Associated With Negative Sentiment")
        if ranked.empty:
            st.dataframe(
                pd.DataFrame(columns=["Concept", "Score", "Pos", "Neg"]),
                use_container_width=True,
            )
        else:
            st.dataframe(
                ranked.tail(15).iloc[::-1][["Concept", "Score", "Pos", "Neg"]],
                use_container_width=True,
            )

    st.header("Deep Dive into Concepts")
    if ranked.empty:
        return

    concepts = ranked["Concept"].dropna().unique().tolist()
    selected = st.selectbox("Select a concept to see related articles:", options=concepts)
    if not selected:
        return

    st.subheader(f"Articles related to '{selected}'")
    pattern = concept_to_pattern(concept_mode, selected)
    text = build_concept_text(filtered)
    matches = filtered[text.str.contains(pattern, case=False, na=False, regex=True)]
    st.dataframe(
        matches[["published_at", "title", "source", "sentiment_compound"]].reset_index(drop=True),
        use_container_width=True,
    )


_download_nltk_resources()
render_app(load_data(str(DATA_PATH)))
