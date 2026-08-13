from dataclasses import dataclass, field


@dataclass
class CategoriesFile:
    """Contents of a `categories.yaml` file.

    Maps each category name to the canonical words it searches for. Those
    words must exist in the synonym file.

        categories:
          Broad Foundational Search:
            - Foundation model
            - Machine Learning
    """

    categories: dict[str, list[str]]


@dataclass
class PaperDiscoveryConfig:
    """Settings for one Paper Discovery run.

    See docs/source/core_features/paper_discovery.md for the full guide.
    """

    # Inputs.
    synonyms_path: str
    """`.jsonl` file, one `{"word": ..., "synonyms": [...]}` object per line."""

    categories_path: str
    """`categories.yaml` file. See `CategoriesFile`."""

    output_file: str
    """Where to write the results, one `Paper` per line. Use a `.jsonl` name."""

    sources: list[str] = field(
        default_factory=lambda: ["openalex", "europepmc", "arxiv"]
    )
    """Which adapters to query. Add `google_scholar` to opt in."""

    # Search limits.
    max_pages: int = 3
    """Paginated requests per source per query."""

    max_results: int = 50
    """Hard cap on results per source per query."""

    user_agent: str = "mmore-paper-discovery/1.0 (https://github.com/EPFLiGHT/mmore)"
    """Sent on every outbound request. Identify yourself so a source can get
    in touch, e.g. `"my-lab-pipeline/1.0 (mailto:alice@example.com)"`.
    OpenAlex gives faster responses to callers with a contact address."""

    arxiv_category_map: dict[str, str] | None = None
    """Substring of a category title to an arXiv code, e.g.
    `{"Foundational": "cs.LG"}`. Adds `cat:<code>` to matching arXiv queries."""

    arxiv_enable_pair_query: bool = True
    """Run one extra arXiv query requiring the top two terms together. Better
    precision, one more round-trip per category."""

    # PDF handling.
    download_pdfs: bool = True
    """Set False to keep metadata and abstracts only."""

    pdf_dir: str = "./pdf_cache"
    """Where downloaded PDFs are cached and reused across runs."""

    force_redownload: bool = False
    """Ignore the cache and re-fetch every PDF."""

    pdf_extractor: str = "fast"
    """Which `mmore.process.PDFProcessor` path to use. `"fast"` is
    PyMuPDF-backed and loads no models. `"full"` is the marker and surya
    pipeline, better on complex layouts but it downloads weights and wants
    a GPU."""

    pdf_proxy_prefix: str | None = None
    """EZproxy host that wraps every PDF URL, e.g.
    `"https://ezproxy.example.edu"`. Only set this if your institution runs
    EZproxy, and use the host your library publishes. Leave unset if access
    comes from VPN and IP recognition, since the direct URL already works."""

    # Extra output.
    multimodal_output_file: str | None = None
    """If set, also write the results as `MultimodalSample` JSONL, which
    `mmore` post-process, index and rag can read without re-processing."""
