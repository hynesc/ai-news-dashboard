from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_SEARCH_QUERY = (
    # High-precision core phrases:
    '"AI safety" OR "AI alignment" OR "AI governance" OR "responsible AI" OR '
    '"frontier model" OR "EU AI Act" OR '
    # Broader governance-adjacent coverage, gated by an AI anchor to reduce noise:
    '((AI OR "artificial intelligence" OR "generative AI") AND '
    "(regulation OR policy OR governance OR oversight OR standards OR "
    'copyright OR "fair use" OR "training data" OR '
    "deepfake OR misinformation OR disinformation OR "
    "privacy OR surveillance OR biometric OR "
    "bias OR discrimination OR "
    "antitrust OR "
    "lawsuit OR court OR "
    '"export controls" OR compute OR gpu OR '
    "security OR cyber OR malware))"
)


@dataclass(frozen=True)
class Settings:
    newsapi_key: Optional[str]
    search_query: str = DEFAULT_SEARCH_QUERY
    output_csv: Path = Path("news_data.csv")

    language: str = "en"
    sort_by: str = "relevancy"
    page_size: int = 100
    max_pages: int = 1

    @staticmethod
    def from_env() -> Settings:
        return Settings(
            newsapi_key=os.getenv("NEWSAPI_KEY"),
            search_query=os.getenv("NEWS_QUERY", DEFAULT_SEARCH_QUERY),
            output_csv=Path(os.getenv("NEWS_OUTPUT_CSV", "news_data.csv")),
        )
