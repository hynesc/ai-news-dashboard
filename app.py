import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import nltk

# --- Function to download NLTK data ---
@st.cache_resource
def download_nltk_resources():
    """Downloads all necessary NLTK resources if they don't exist."""
    resources = ["stopwords", "punkt"]
    for resource in resources:
        try:
            # A robust way to check if the data is present
            nltk.data.find(f"corpora/{resource}.zip" if resource == 'stopwords' else f"tokenizers/{resource}")
        except LookupError:
            print(f"Downloading NLTK resource: {resource}...")
            nltk.download(resource, quiet=True)

# --- Call the downloader at the start ---
download_nltk_resources()


# --- Page Configuration ---
st.set_page_config(
    page_title="AI Safety & Governance Tracker",
    page_icon="🤖",
    layout="wide"
)

# --- Load Data ---
@st.cache_data(ttl=600)
def load_data(filename):
    """Loads data from CSV and caches it."""
    try:
        df = pd.read_csv(filename, parse_dates=['published_at'])
        df['published_at'] = df['published_at'].dt.tz_localize(None)
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file '{filename}' was not found. Please run the data collection script.")
        return pd.DataFrame()

data = load_data("news_data.csv")

# --- Helper Functions ---
def get_ngrams_df(texts, ngram_range=(2, 3), n_top=10):
    """Generates a DataFrame of top n-grams."""
    if texts.empty:
        return pd.DataFrame(columns=['Concept', 'Frequency'])
    try:
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
        X = vectorizer.fit_transform(texts)
        sum_words = X.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return pd.DataFrame(words_freq[:n_top], columns=['Concept', 'Frequency'])
    except ValueError:
        return pd.DataFrame(columns=['Concept', 'Frequency'])

def create_wordcloud(texts):
    """Generates and returns a word cloud figure."""
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(['ai'])    # Manually add 'ai' as a stop word
    
    text = " ".join(texts)
    wordcloud = WordCloud(
        width=800, height=400, background_color='white', stopwords=stop_words
    ).generate(text)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

# --- App Title ---
st.title("🤖 AI Safety & Governance Tracker")
st.markdown("An automated dashboard tracking global discourse on AI safety, alignment, and policy.")

# --- Main Dashboard ---
if data.empty:
    st.warning("No data to display. The data file is missing or empty.")
else:
    st.sidebar.header("Filters")
    timeframe_options = {
        "Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90, "All Time": None
    }
    selected_timeframe_key = st.sidebar.selectbox("Select Timeframe", options=list(timeframe_options.keys()))

    if timeframe_options[selected_timeframe_key] is not None:
        days = timeframe_options[selected_timeframe_key]
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
        filtered_data = data[data['published_at'] >= cutoff_date]
    else:
        filtered_data = data

    if filtered_data.empty:
        st.warning(f"No articles found for the selected timeframe: **{selected_timeframe_key}**.")
    else:
        st.header(f"Analytics for: {selected_timeframe_key}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Articles", f"{len(filtered_data):,}")
        avg_sentiment = filtered_data['sentiment_compound'].mean()
        col2.metric("Avg. Sentiment Score", f"{avg_sentiment:.3f}" if not pd.isna(avg_sentiment) else "N/A")
        col3.metric("Unique News Sources", f"{filtered_data['source'].nunique():,}")

        st.header("Visual Analysis")
        tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Trend", "Sentiment Distribution", "Sentiment by Source", "Word Cloud"])

        with tab1:
            st.subheader("Sentiment Over Time")
            sentiment_over_time = filtered_data.set_index('published_at').resample('D')['sentiment_compound'].mean().dropna()
            fig = px.line(sentiment_over_time, labels={"value": "Avg. Sentiment", "index": "Date"}, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.subheader("Distribution of Sentiment Scores")
            fig = px.histogram(filtered_data, x="sentiment_compound", nbins=50, title="Distribution of Sentiment Scores")
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            st.subheader("Average Sentiment by Top Sources")
            sentiment_by_source = filtered_data.groupby('source')['sentiment_compound'].agg(['mean', 'count']).rename(columns={'mean':'avg_sentiment', 'count':'article_count'}).nlargest(15, 'article_count')
            fig = px.bar(sentiment_by_source, y='avg_sentiment', title="Avg. Sentiment for Top 15 Sources (by article volume)")
            st.plotly_chart(fig, use_container_width=True)
        with tab4:
            st.subheader("Headline Word Cloud")
            fig = create_wordcloud(filtered_data['title'].dropna())
            st.pyplot(fig)

        st.header("Key Concept Analysis")
        positive_articles = filtered_data[filtered_data['sentiment_compound'] > 0.1]
        negative_articles = filtered_data[filtered_data['sentiment_compound'] < -0.1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Concepts in Positive News")
            pos_ngrams_df = get_ngrams_df(positive_articles['title'].dropna())
            st.dataframe(pos_ngrams_df, use_container_width=True)
        with col2:
            st.subheader("Top Concepts in Negative News")
            neg_ngrams_df = get_ngrams_df(negative_articles['title'].dropna())
            st.dataframe(neg_ngrams_df, use_container_width=True)

        st.header("Deep Dive into Concepts")
        all_concepts = pd.concat([pos_ngrams_df, neg_ngrams_df])['Concept'].dropna().unique().tolist()
        
        if all_concepts:
            selected_concept = st.selectbox("Select a concept to see related articles:", options=all_concepts)
            if selected_concept:
                st.subheader(f"Articles related to '{selected_concept}'")
                
                search_terms = selected_concept.split()
                pattern = "".join([f"(?=.*{re.escape(term)})" for term in search_terms])
                
                concept_articles = filtered_data[filtered_data['title'].str.contains(pattern, case=False, na=False, regex=True)]
                
                st.dataframe(concept_articles[['published_at', 'title', 'source', 'sentiment_compound']], use_container_width=True)
        else:
            st.info("No concepts generated from the selected data to perform a deep dive.")
