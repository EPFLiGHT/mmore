from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

SourceName = Literal["arxiv", "openalex", "europepmc", "google_scholar"]


@dataclass
class CategoryQuery:
    """Boolean query for one category - output of Stage 1, input to Stage 2."""

    combination_title: str
    boolean_combination: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "combination_title": self.combination_title,
            "boolean_combination": self.boolean_combination,
        }


@dataclass
class Paper:
    """One academic paper, normalized across all sources."""

    title: Optional[str] = None
    authors: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    year: Optional[int] = None
    extracted_text: Optional[str] = None
    source: Optional[SourceName] = None
    search_category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "abstract": self.abstract,
            "year": self.year,
            "extracted_text": self.extracted_text,
            "source": self.source,
            "search_category": self.search_category,
        }


@dataclass
class SynonymEntry:
    """One row of the synonym table."""

    word: str
    synonyms: List[str] = field(default_factory=list)
