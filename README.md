# AI Safety & Governance News Tracker

A live, automated dashboard that tracks and analyzes global news discourse on the critical topics of AI safety, alignment, ethics, and policy.

## Live Dashboard

You can view the deployed interactive dashboard here:
**[https://ai-news-dashboard.streamlit.app/](https://ai-news-dashboard.streamlit.app/)**

## Features

This dashboard provides a real-time view of the global conversation around AI safety and governance, with features including:

- **Automated Daily Updates:** A data pipeline runs every 24 hours to fetch the latest news, ensuring the data is always current.
- **Sentiment Analysis:** Each article is analyzed using NLTK's VADER to determine the sentiment (Positive, Neutral, Negative) of the discourse. **Note:** Due to API limitations on the free tier, analysis is currently only performed on the combined article title and content snippet.
- **Trend Tracking:** An interactive chart visualizes how the average sentiment has changed over time.
- **Key Concept Identification:** An n-gram analysis identifies the most frequent and important multi-word concepts being discussed in headlines.
- **Sentiment-Based Insights:** The dashboard separates key concepts found in positive news from those in negative news, revealing what topics are driving different narratives.
- **Interactive Deep Dive:** Users can select a concept to instantly filter and view all related articles.
- **Data Filtering:** All analytics can be filtered by time period (Last 7 Days, 30 Days, 90 Days, All Time).

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
5.  **Dashboard Refresh:** The live Streamlit app detects the change in the data file and automatically refreshes its cache, ensuring the dashboard always displays the most current information.

## Development

This repo keeps `app.py` and `news_pipeline.py` as stable entrypoints for Streamlit Cloud and GitHub Actions.
Shared logic lives under `src/ai_news_dashboard/`.

### Run The Dashboard

```bash
streamlit run app.py
```

### Run The Pipeline Locally

```bash
export NEWSAPI_KEY="..."
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
