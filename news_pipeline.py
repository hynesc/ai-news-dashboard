import os
import re
import pandas as pd
import nltk
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Dict, Any

# --- 1. Centralized Configuration ---
# API key provided by the GitHub Actions secret
API_KEY = os.getenv('NEWSAPI_KEY')
SEARCH_QUERY = (
    '"AI safety" OR "AI alignment" OR "AI governance" OR "responsible AI" OR "AGI" OR "EU AI Act" OR '
    '("existential risk" AND AI) OR (regulation AND AI) OR (ethics AND AI) OR (bias AND AI) OR '
    '(("misinformation" OR "disinformation") AND AI) OR (copyright AND AI) OR '
    '("red-teaming" AND AI) OR (interpretability AND AI)'
)
OUTPUT_FILENAME = "news_data.csv"

# --- 2. Encapsulate Logic in a Class ---
class NewsProcessor:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key must be provided.")
        self._download_nltk_resources()
        self.api = NewsApiClient(api_key=api_key)
        self.sid = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['ai', 'artificial', 'intelligence', 'agi'])

    def _download_nltk_resources(self):
        resources = ["vader_lexicon", "punkt", "wordnet", "stopwords"]
        for resource in resources:
            try:
                nltk.data.find(f"corpora/{resource}.zip" if resource == 'stopwords' else f"tokenizers/{resource}")
            except LookupError:
                print(f"Downloading NLTK resource: {resource}...")
                nltk.download(resource, quiet=True)

    def fetch_articles(self, query: str) -> List[Dict[str, Any]]:
        print(f"Fetching news for query: '{query}'...")
        try:
            response = self.api.get_everything(q=query, language="en", sort_by="relevancy", page_size=100)
            return response.get('articles', [])
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

    def analyze_sentiment(self, articles: List[Dict[str, Any]]) -> pd.DataFrame:
        processed_data = []
        for article in articles:
            title = article.get('title', '')
            description = article.get('description') or article.get('content', '')
            text_for_sentiment = f"{title}. {description}"
            if not text_for_sentiment.strip():
                continue
            sentiment_scores = self.sid.polarity_scores(text_for_sentiment)
            processed_data.append({
                'source': article.get('source', {}).get('name'),
                'title': title, 'url': article.get('url'),
                'published_at': pd.to_datetime(article.get('publishedAt')),
                'content_snippet': description, 'sentiment_compound': sentiment_scores['compound'],
            })
        return pd.DataFrame(processed_data)

    def get_top_ngrams(self, texts: pd.Series, n_gram_range: tuple = (2, 3), n_top: int = 15) -> None:
        print(f"\n--- Top {n_top} Concepts (Bi-grams & Tri-grams) ---")
        try:
            vectorizer = CountVectorizer(ngram_range=n_gram_range, stop_words='english')
            X = vectorizer.fit_transform(texts)
            sum_words = X.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            
            for word, freq in words_freq[:n_top]:
                print(f"  - {word} (occurs {freq} times)")
        except ValueError:
            print("  Not enough data to generate n-grams.")


def main():
    if not API_KEY:
        print("Error: NEWSAPI_KEY environment variable not set.")
        return
        
    processor = NewsProcessor(api_key=API_KEY)
    
    try:
        existing_df = pd.read_csv(OUTPUT_FILENAME, parse_dates=['published_at'])
    except FileNotFoundError:
        existing_df = pd.DataFrame()

    new_articles = processor.fetch_articles(query=SEARCH_QUERY)
    if not new_articles:
        print("No new articles fetched.")
    else:
        new_df = processor.analyze_sentiment(new_articles)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['title'], keep='last', inplace=True)
        combined_df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8')
        print(f"Saved {len(combined_df)} total articles to {OUTPUT_FILENAME}")
        
        processor.get_top_ngrams(texts=combined_df['title'].dropna())


if __name__ == "__main__":
    main()
