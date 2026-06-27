from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PaperDiscoveryConfig:
    """Configuration for the Paper Discovery pipeline.

    Fields:
      synonyms_path:     Path to a JSON file containing a list of
                         {"word": str, "synonyms": [str, ...]} entries.
      categories:        Mapping of category name -> list of `word` keys
                         from the synonym file.
      sources:           Which source adapters to enable.
      output_file:       Where to write the final papers.json.
      download_pdfs:     If True, download PDFs and extract text.
      pdf_dir:           Directory under which downloaded PDFs are cached.
      max_pages:         Max paginated requests per source per query.
      max_results:       Hard cap on results returned per source per query.
      user_agent:        HTTP `User-Agent` header sent on every outbound
                         request to source APIs (OpenAlex, Europe PMC,
                         arXiv) and to publisher PDF endpoints. "Polite"
                         here means a string that identifies the caller
                         honestly so rate-limiters / abuse desks can
                         contact you - e.g.
                         `"my-lab-pipeline/1.0 (mailto:alice@example.com)"`.
                         OpenAlex in particular routes UAs with a contact
                         address into a faster, more reliable pool.
      pdf_proxy_prefix:  Optional EZproxy-style prefix that wraps every PDF
                         URL for institutional access (e.g.
                         "https://login.proxy.epfl.ch"). Leave None to fetch
                         URLs directly — expect 403s on paywalled publishers.
    """

    synonyms_path: str
    categories: Dict[str, List[str]]
    output_file: str
    sources: List[str] = field(
        default_factory=lambda: ["openalex", "europepmc", "arxiv"]
    )
    download_pdfs: bool = True
    pdf_dir: str = "./pdf_cache"
    max_pages: int = 3
    max_results: int = 50
    user_agent: str = "mmore-paper-discovery/1.0 (https://github.com/swiss-ai/mmore)"
    arxiv_category_map: Optional[Dict[str, str]] = None
    pdf_proxy_prefix: Optional[str] = None
    force_redownload: bool = False  # If True, ignore the on-disk PDF cache.
