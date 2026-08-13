"""Source adapter registry.

Each entry in `REGISTRY` maps a config-facing name (e.g. `"arxiv"`) to the
adapter class that implements `SourceAdapter`. `get_adapter()` instantiates
one with the constructor kwargs forwarded from `PaperDiscoveryConfig`.

To add a new source: implement the `SourceAdapter` protocol in a new module
and register it here.

Google Scholar is registered unconditionally. Its adapter imports
`scholarly` lazily inside `search()`, so a missing package logs a warning
and returns nothing rather than breaking the import.
"""

from .arxiv import ArxivAdapter
from .base import SourceAdapter
from .europepmc import EuropePmcAdapter
from .google_scholar import GoogleScholarAdapter
from .openalex import OpenAlexAdapter

REGISTRY: dict[str, type[SourceAdapter]] = {
    "openalex": OpenAlexAdapter,
    "europepmc": EuropePmcAdapter,
    "arxiv": ArxivAdapter,
    "google_scholar": GoogleScholarAdapter,
}


def get_adapter(name: str, **kwargs) -> SourceAdapter:
    """Instantiate the adapter registered under `name`.

    Args:
      name: Source key from `PaperDiscoveryConfig.sources`, such as
        `"openalex"` or `"arxiv"`.
      kwargs: Passed to the adapter constructor. Usually `user_agent`,
        `max_pages` and `max_results`, plus source-specific extras such as
        arXiv's `category_map` and `enable_pair_query`.

    Raises:
      ValueError: if `name` is not in `REGISTRY`.
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown source: {name!r}. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
