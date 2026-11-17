from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import STOPWORDS, WordCloud

src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ai_news_dashboard.analytics import (  # noqa: E402
    build_concept_text,
    cap_articles_per_source,
    compute_ranked_concepts,
    concept_to_pattern,
    filter_by_timeframe,
    normalize_data,
    sentiment_trend,
    split_sentiment_buckets,
)

DATA_PATH = Path(__file__).resolve().parent / "news_data.csv"
TIMEFRAMES = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "All time": None}


@st.cache_data(show_spinner=False, max_entries=4)
def load_data(path: str, file_version: tuple[int, int]) -> pd.DataFrame:
    # File metadata is part of the cache key, so new pipeline output appears immediately.
    return normalize_data(pd.read_csv(path))


@st.cache_data(show_spinner=False, max_entries=16)
def wordcloud_image(headlines: tuple[str, ...]) -> bytes:
    cloud = WordCloud(
        width=1200,
        height=500,
        background_color="white",
        stopwords=STOPWORDS | {"ai"},
        random_state=42,
    ).generate(" ".join(headlines))
    output = io.BytesIO()
    cloud.to_image().save(output, format="PNG")
    return output.getvalue()


@st.cache_data(show_spinner=False, max_entries=32)
def ranked_concepts(**kwargs) -> pd.DataFrame:
    return compute_ranked_concepts(**kwargs)


def article_table(df: pd.DataFrame, key: str) -> None:
    st.dataframe(
        df[["published_at", "title", "source", "sentiment_compound", "url"]],
        hide_index=True,
        width="stretch",
        key=key,
        column_config={
            "published_at": st.column_config.DatetimeColumn(
                "Published (UTC)", format="MMM D, YYYY HH:mm"
            ),
            "title": st.column_config.TextColumn("Headline", width="large"),
            "source": "Source",
            "sentiment_compound": st.column_config.NumberColumn("Sentiment", format="%.3f"),
            "url": st.column_config.LinkColumn("Article", display_text="Read article"),
        },
    )
    st.caption(f"{len(df):,} articles · newest first · timestamps in UTC")


def render_app() -> None:
    st.set_page_config(page_title="AI Safety & Governance Tracker", page_icon="🤖", layout="wide")
    st.title("AI Safety & Governance Tracker")
    st.write("Explore the news shaping AI safety, alignment, and policy.")
    try:
        stat = DATA_PATH.stat()
        df = load_data(str(DATA_PATH), (stat.st_mtime_ns, stat.st_size))
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError):
        st.error("The news archive is temporarily unavailable. Please try again later.")
        return
    if df.empty:
        st.info("No articles have been collected yet. Please check back later.")
        return

    latest = df["published_at"].max()
    if pd.notna(latest):
        st.caption(
            f"Latest article: {latest:%B %d, %Y at %H:%M UTC} · {len(df):,} articles in archive"
        )
        age = (pd.Timestamp.now(tz="UTC") - latest).days
        if age > 3:
            st.warning(
                f"The newest article is {age:,} days old. Coverage may not reflect recent events."
            )

    st.sidebar.header("Explore coverage")
    timeframe = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES), index=3)
    query = st.sidebar.text_input("Search articles", placeholder="Search headlines and snippets")
    sources = st.sidebar.multiselect("Sources", sorted(df["source"].unique()))
    sentiment_filter = st.sidebar.multiselect(
        "Sentiment", ["Positive", "Neutral", "Negative", "Unscored"]
    )

    filtered = filter_by_timeframe(df, TIMEFRAMES[timeframe])
    if query.strip():
        filtered = filtered[
            build_concept_text(filtered).str.contains(query.strip(), case=False, regex=False)
        ]
    if sources:
        filtered = filtered[filtered["source"].isin(sources)]
    if sentiment_filter:
        scores = filtered["sentiment_compound"]
        labels = pd.Series("Neutral", index=filtered.index)
        labels[scores > 0.05] = "Positive"
        labels[scores < -0.05] = "Negative"
        labels[scores.isna()] = "Unscored"
        filtered = filtered[labels.isin(sentiment_filter)]
    st.sidebar.caption(
        "Sentiment measures language tone, not whether an event is good or bad for AI safety."
    )
    if filtered.empty:
        st.info(
            "No articles match these filters. Try a longer timeframe or clear search, "
            "source, and sentiment filters."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles", f"{len(filtered):,}")
    c2.metric("Sources", f"{filtered['source'].nunique():,}")
    avg = filtered["sentiment_compound"].mean()
    c3.metric("Average sentiment", f"{avg:+.3f}" if pd.notna(avg) else "N/A")
    c4.metric("Scored articles", f"{filtered['sentiment_compound'].notna().sum():,}")

    overview, concepts, articles = st.tabs(
        ["Coverage overview", "Concept analysis", "Browse articles"]
    )
    with overview:
        chart = st.selectbox(
            "View",
            ["Sentiment over time", "Sentiment distribution", "Top sources", "Headline word cloud"],
        )
        if chart == "Sentiment over time":
            trend = sentiment_trend(filtered)
            if trend.empty or trend["Daily mean"].isna().all():
                st.info("Dated, scored articles are needed to show a trend.")
            else:
                fig = px.line(
                    trend,
                    x="published_at",
                    y=["Daily mean", "7-day moving average"],
                    labels={
                        "value": "Sentiment",
                        "published_at": "Date (UTC)",
                        "variable": "Series",
                    },
                )
                fig.update_traces(
                    selector={"name": "Daily mean"}, opacity=0.4, mode="lines+markers"
                )
                fig.update_traces(selector={"name": "7-day moving average"}, line={"width": 3})
                fig.add_hline(y=0, line_dash="dot", opacity=0.4)
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "Daily mean and a trailing 7-calendar-day average. "
                    "Gaps indicate days without scored articles."
                )
        elif chart == "Sentiment distribution":
            fig = px.histogram(
                filtered,
                x="sentiment_compound",
                nbins=40,
                labels={"sentiment_compound": "Sentiment score"},
            )
            st.plotly_chart(fig, width="stretch")
        elif chart == "Top sources":
            by_source = (
                filtered.groupby("source")
                .agg(average_sentiment=("sentiment_compound", "mean"), articles=("title", "size"))
                .nlargest(15, "articles")
                .sort_values("articles")
                .reset_index()
            )
            fig = px.bar(
                by_source,
                x="articles",
                y="source",
                orientation="h",
                color="average_sentiment",
                color_continuous_scale="RdBu",
                range_color=[-1, 1],
                hover_data={"average_sentiment": ":.3f"},
                labels={
                    "articles": "Articles",
                    "source": "Source",
                    "average_sentiment": "Sentiment",
                },
            )
            st.plotly_chart(fig, width="stretch")
        else:
            try:
                st.image(wordcloud_image(tuple(filtered["title"])), width="stretch")
            except ValueError:
                st.info("There are no usable headline words in this selection.")
        st.caption(
            "VADER scores range from −1 to +1. Positive: > 0.05 · Neutral: −0.05 to 0.05 · "
            "Negative: < −0.05. Scores use headlines and snippets, not full articles."
        )

    with concepts:
        st.subheader("What distinguishes positive and negative coverage?")
        with st.expander("Analysis settings"):
            mode = st.selectbox("Concept method", ["Topics (general)", "Phrases (ngrams)"])
            quantile = st.slider(
                "Sentiment tail size",
                0.10,
                0.40,
                0.20,
                0.05,
                help="Compare the highest and lowest scoring tails; neutral articles are excluded.",
            )
            cap = st.number_input(
                "Max articles per source in each bucket",
                0,
                50,
                10,
                help="Use 0 to include all articles.",
            )
            min_df, min_hits, min_total = 2, 2, 3
            if mode == "Phrases (ngrams)":
                min_df = st.number_input("Min phrase frequency", 1, 20, 2)
                min_hits = st.number_input("Min phrase hits in one bucket", 1, 20, 2)
                min_total = st.number_input("Min phrase hits overall", 1, 50, 3)
        pos, neg = split_sentiment_buckets(filtered, quantile)
        pos, neg = cap_articles_per_source(pos, cap), cap_articles_per_source(neg, cap)
        st.caption(
            f"Comparing {len(pos):,} positive and {len(neg):,} negative articles "
            "after source caps. "
            "Scores are smoothed log odds: above zero favors positive coverage; "
            "below zero favors negative coverage."
        )
        if pos.empty or neg.empty:
            st.info(
                "Both positive and negative articles are needed. "
                "Broaden your filters to compare concepts."
            )
        else:
            ranked = ranked_concepts(
                concept_mode=mode,
                positive_articles=pos,
                negative_articles=neg,
                ngram_min_df=min_df,
                ngram_min_hits=min_hits,
                ngram_min_total=min_total,
            )
            if ranked.empty:
                st.info(
                    "No concepts meet these thresholds. "
                    "Try a longer timeframe or lower phrase thresholds."
                )
            else:
                c1, c2 = st.columns(2)
                for col, label, subset in [
                    (c1, "Associated with positive coverage", ranked[ranked["Score"] > 0].head(15)),
                    (
                        c2,
                        "Associated with negative coverage",
                        ranked[ranked["Score"] < 0].tail(15).iloc[::-1],
                    ),
                ]:
                    col.markdown(f"**{label}**")
                    col.dataframe(
                        subset,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "Score": st.column_config.NumberColumn(format="%.2f"),
                            "Pos": "Positive hits",
                            "Neg": "Negative hits",
                        },
                    )
                selected = st.selectbox("Explore a concept", ranked["Concept"].tolist())
                matches = filtered[
                    build_concept_text(filtered).str.contains(
                        concept_to_pattern(mode, selected), case=False, na=False, regex=True
                    )
                ]
                article_table(matches, "concept_articles")

    with articles:
        st.subheader("Article archive")
        st.write(
            "Search and narrow coverage using the sidebar. Open an original article from the table."
        )
        article_table(filtered, "all_articles")
        st.download_button(
            "Download filtered articles (CSV)",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="ai-news-filtered.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    render_app()
