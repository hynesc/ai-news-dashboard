# AI Safety & Governance News Tracker

An interactive news archive tracking coverage of AI safety, alignment, ethics, and policy. The dashboard shows the newest article date and warns when coverage is stale.

## Live Dashboard

You can view the deployed interactive dashboard here:
**[https://ai-news-dashboard.streamlit.app/](https://ai-news-dashboard.streamlit.app/)**

## Features

Explore collected coverage of AI safety and governance with:

- **Scheduled Collection:** A daily pipeline collects news when GitHub Actions and the NewsAPI credentials are active. The dashboard makes stale coverage visible.
- **Sentiment Analysis:** Each article is analyzed using NLTK's VADER to determine the sentiment (Positive, Neutral, Negative) of the discourse. **Note:** Due to API limitations on the free tier, analysis is currently only performed on the combined article title and content snippet.
- **Trend Tracking:** An interactive chart visualizes how the average sentiment has changed over time.
- **Key Concept Identification:** Choose curated topics or extracted phrases from headlines and snippets. Both methods compare article-level occurrence using smoothed log odds, normalized for sentiment bucket sizes.
- **Sentiment-Based Insights:** The dashboard compares the highest and lowest sentiment tails, excludes neutral articles, and caps source representation. Sentiment describes language tone, not the impact of events on AI safety.
- **Article Browser:** Search headlines and snippets, filter sources and sentiment, open original articles, and download the filtered dataset as CSV. Concept selections also link to related coverage.
- **Data Filtering:** Filter all views by timeframe, source, search text, and sentiment. Dates use UTC; articles without valid dates remain available in All time.

## Tech Stack & Methodology

This project leverages a combination of data science and software engineering practices to create a robust analysis pipeline.


- **Data Collection:** The [NewsAPI](https://newsapi.org/) is used to fetch relevant news articles. The dataset is built using a highly specific, strategic search query that targets key phrases such as:
    - **Core Phrases:** `"AI safety"`, `"AI alignment"`, `"AI governance"`, `"responsible AI"`, `"AGI"`, `"EU AI Act"`
    - **Associated Concepts:** `existential risk`, `regulation`, `ethics`, `bias`, `misinformation`, `disinformation`, `copyright`, `red-teaming`, and `interpretability` when mentioned with AI.
- **Data Processing & Analysis:**
    - **Pandas:** For data manipulation and aggregation.
    - **NLTK (VADER):** For nuanced sentiment analysis tuned for news and social media text.
    - **Scikit-learn:** For n-gram feature extraction to identify key concepts.
- **Frontend Dashboard:**
    - **Streamlit:** For creating and deploying the interactive web application.
    - **Plotly Express:** For generating interactive data visualizations.
    - **WordCloud:** For visual representation of word frequencies.
- **Automation & Deployment:**
    - **GitHub Actions:** For scheduling and running the daily data collection script.
    - **Streamlit Community Cloud:** For hosting the live web application.

## Automated Data Pipeline

The core of this project is an automated data pipeline managed by GitHub Actions:

1.  **Scheduled Run:** A CRON job triggers the workflow every 24 hours.
2.  **Data Fetching:** The `news_pipeline.py` script executes, calling the NewsAPI to fetch the latest 100 articles matching the strategic query.
3.  **Analysis & Aggregation:** The script analyzes the sentiment of the new articles, appends them to the existing `news_data.csv`, removes duplicates, and saves the updated file.
4.  **Commit to Repo:** The GitHub Action automatically commits the updated `news_data.csv` back to the repository.
5.  **Dashboard Refresh:** On the next interaction or page load, a file metadata cache key detects updated CSV data. Failed API requests fail the workflow, and atomic file replacement protects the archive from interrupted writes.

## Development

This repo keeps `app.py` and `news_pipeline.py` as stable entrypoints for Streamlit Cloud and GitHub Actions.
Shared logic lives under `src/ai_news_dashboard/`.

### Run The Dashboard

Use Python 3.9+ and install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Run The Pipeline Locally

```bash
export NEWSAPI_KEY="..."
export NEWS_QUERY='("AI safety" OR ... )'   # optional override
export NEWS_OUTPUT_CSV="news_data.csv"      # optional override
python news_pipeline.py
```

### Code Quality (Optional)

Dev tooling is configured in `pyproject.toml`.

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .
ruff format .
pytest
```

### Analysis and performance notes

The trend uses a trailing seven-calendar-day average, including gaps in coverage.
Topic and phrase scores above zero favor positive coverage; scores below zero favor negative
coverage. Counts are articles, not repeated word mentions. Phrase thresholds are respected;
empty results suggest broadening filters or lowering thresholds. These comparisons are descriptive,
not statistical significance tests.

The dashboard runs without NLTK dataset downloads. Word clouds are generated only when selected,
and their PNGs and concept rankings use bounded caches. The collection pipeline still needs the
VADER lexicon and downloads it if missing. Tests cover data normalization, analytics edge cases,
pipeline failures and atomic writes, plus Streamlit views and filter interactions.
