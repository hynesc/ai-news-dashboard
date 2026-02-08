from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional


DEFAULT_SEARCH_QUERY = (
    '"AI safety" OR "AI alignment" OR "AI governance" OR "responsible AI" OR "AGI" OR "EU AI Act" OR '
    '("existential risk" AND AI) OR (regulation AND AI) OR (ethics AND AI) OR (bias AND AI) OR '
    '(("misinformation" OR "disinformation") AND AI) OR (copyright AND AI) OR '
    '("red-teaming" AND AI) OR (interpretability AND AI)'
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
    def from_env() -> "Settings":
        return Settings(newsapi_key=os.getenv("NEWSAPI_KEY"))
