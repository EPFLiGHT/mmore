from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..type import MultimodalSample


class SourceName(str, Enum):
    """Which repository a `Paper` came from.

    Subclasses `str` so members serialize as their plain value, `"arxiv"`
    rather than `"SourceName.ARXIV"`, with no custom JSON encoder.
    """

    ARXIV = "arxiv"
    OPENALEX = "openalex"
    EUROPEPMC = "europepmc"
    GOOGLE_SCHOLAR = "google_scholar"


@dataclass
class CategoryQuery:
    """A boolean query for one category. Stage 1 builds these, stage 2 runs them."""

    combination_title: str
    boolean_combination: str

    def to_dict(self) -> dict[str, str]:
        return {
            "combination_title": self.combination_title,
            "boolean_combination": self.boolean_combination,
        }


@dataclass
class Paper:
    """One paper, in the same shape whichever source it came from.

    Every field is optional because sources differ in what they return.
    `None` means we do not know, not that the value is empty.
    """

    title: str | None = None
    authors: list[str] | None = None
    url: str | None = None
    abstract: str | None = None
    year: int | None = None
    extracted_text: str | None = None
    source: SourceName | None = None
    search_category: str | None = None

    def to_dict(self) -> dict[str, Any]:
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
        """Convert to mmore's document shape so index and rag can read it.

        Args:
          pdf_path: Path to the cached PDF, if one was downloaded.

        Returns:
          A `MultimodalSample` whose text is the extracted PDF body, falling
          back to the abstract and then the title. The paper's own fields go
          in `metadata.extra` so nothing is lost through JSONL.
        """
        # Imported here rather than at module scope to keep this module
        # cheap to import when only the dataclasses are needed.
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
    """A canonical word and the terms that mean the same thing."""

    word: str
    synonyms: list[str] = field(default_factory=list)
