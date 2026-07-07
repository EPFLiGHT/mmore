from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from ..type import MultimodalSample

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
    authors: Optional[List[str]] = None
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

    def to_multimodal_sample(self, pdf_path: str = "") -> "MultimodalSample":
        """Convert this Paper into a `mmore.type.MultimodalSample`.

        The result plugs directly into mmore's downstream pipelines
        (post-process -> index -> rag) so a Paper Discovery run does
        not need a second pass through `mmore process`.

        - `text` uses the extracted PDF body if available, otherwise
          the abstract, otherwise the title.
        - `metadata.file_path` points at the cached PDF when known.
        - Paper-specific fields (title, authors, year, source, url,
          search_category, abstract) live under `metadata.extra` so
          they survive the JSONL round-trip.
        """
        # Local import - mmore.type is core, but keeping the schema
        # module free of heavy imports at load time is still cheaper.
        from ..type import DocumentMetadata, MultimodalSample

        body = self.extracted_text or self.abstract or self.title or ""
        extra = {
            k: v
            for k, v in {
                "title": self.title,
                "authors": self.authors,
                "year": self.year,
                "source": self.source,
                "url": self.url,
                "search_category": self.search_category,
                "abstract": self.abstract,
            }.items()
            if v is not None
        }
        return MultimodalSample(
            text=body,
            modalities=[],
            metadata=DocumentMetadata(
                file_path=pdf_path,
                processor_type="paper_discovery",
                extra=extra,
            ),
        )


@dataclass
class SynonymEntry:
    """One row of the synonym table."""

    word: str
    synonyms: List[str] = field(default_factory=list)
