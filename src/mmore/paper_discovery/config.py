from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CategoriesFile:
    """Wrapper around `categories.yaml`.

    The YAML file holds a single top-level key `categories` whose value is
    a mapping from human-readable category names to a list of canonical
    `word`s that must exist in the synonym table.

    Example:
        categories:
          Broad Foundational Search:
            - Foundation model
            - Machine Learning
    """

    categories: Dict[str, List[str]]


@dataclass
class PaperDiscoveryConfig:
    """Configuration for the Paper Discovery pipeline.

    Fields:
      synonyms_path:           Path to a `.jsonl` synonym file - one
                               `{"word": str, "synonyms": [...]}` object
                               per line.
      categories_path:         Path to a `categories.yaml` file (see
                               `CategoriesFile`). Maps category names to
                               lists of canonical `word`s from the
                               synonym file.
      sources:                 Which source adapters to enable.
      output_file:             Where to write the final JSONL of
                               `Paper` records (one paper per line).
                               Convention: end the filename in
                               `.jsonl`.
      download_pdfs:           If True, download PDFs and extract text.
      pdf_dir:                 Directory under which downloaded PDFs are
                               cached.
      max_pages:               Max paginated requests per source per query.
      max_results:             Hard cap on results returned per source per
                               query.
      user_agent:              HTTP `User-Agent` header sent on every
                               outbound request to source APIs (OpenAlex,
                               Europe PMC, arXiv) and to publisher PDF
                               endpoints. "Polite" here means a string
                               that identifies the caller honestly so
                               rate-limiters / abuse desks can contact
                               you - e.g.
                               `"my-lab-pipeline/1.0 (mailto:alice@example.com)"`.
                               OpenAlex routes UAs with a contact address
                               into a faster, more reliable pool.
      pdf_proxy_prefix:        Optional EZproxy-style prefix that wraps
                               every PDF URL for institutional access
                               (e.g. "https://login.proxy.epfl.ch").
                               Leave None to fetch URLs directly -
                               expect 403s on paywalled publishers.
      arxiv_category_map:      Substring-of-category-title -> arXiv
                               category code (e.g. "Foundational" ->
                               "cs.LG"). Adds `cat:<code>` to the arXiv
                               query for matching categories.
      arxiv_enable_pair_query: When True (default), the arXiv adapter
                               adds one extra targeted query that ANDs
                               the top two simplified terms. Set False
                               to skip it (saves one round-trip per
                               category).
      pdf_extractor:           Which `mmore.process.PDFProcessor` path
                               to use when extracting text. Two values:
                                 - "fast" (default): PyMuPDF-backed
                                   `process_fast` - no marker/surya
                                   models loaded. Good enough for most
                                   papers.
                                 - "full": the full marker + surya
                                   pipeline used by `mmore process`.
                                   Better layout handling on complex
                                   PDFs; downloads model weights on
                                   first use and wants a GPU to be
                                   fast.
      force_redownload:        If True, ignore the on-disk PDF cache.
      multimodal_output_file:  Optional path. If set, in addition to
                               the `papers.jsonl` output, the pipeline
                               also writes a JSONL of `MultimodalSample`
                               records (mmore's canonical
                               processed-document shape).
                               That file plugs directly into
                               `mmore process` post-processing,
                               `mmore index`, and the RAG pipeline
                               with no re-processing needed.
    """

    synonyms_path: str
    categories_path: str
    output_file: str
    sources: List[str] = field(
        default_factory=lambda: ["openalex", "europepmc", "arxiv"]
    )
    download_pdfs: bool = True
    pdf_dir: str = "./pdf_cache"
    max_pages: int = 3
    max_results: int = 50
    user_agent: str = "mmore-paper-discovery/1.0 (https://github.com/EPFLiGHT/mmore)"
    arxiv_category_map: Optional[Dict[str, str]] = None
    arxiv_enable_pair_query: bool = True
    pdf_extractor: str = "fast"
    pdf_proxy_prefix: Optional[str] = None
    force_redownload: bool = False
    multimodal_output_file: Optional[str] = None
