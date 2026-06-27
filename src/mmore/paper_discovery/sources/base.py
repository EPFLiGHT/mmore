"""Source adapter protocol. Every adapter follows the same shape:

    adapter = MySourceAdapter(user_agent=..., max_pages=..., max_results=...)
    papers = adapter.search(query, category_title)

The constructor kwargs (`user_agent`, `max_pages`, `max_results`, and any
source-specific extras) are passed in by `get_adapter()` at runtime — they
are NOT part of the Protocol surface. The Protocol only pins the public
contract that the pipeline relies on: a `name` attribute and `search()`.

Adapters MUST NOT raise on network errors. Return [] and log instead.
"""

from typing import List, Protocol

from ..schema import Paper


class SourceAdapter(Protocol):
    """Read-only protocol every source adapter satisfies.

    Implementations live in `sources/<source>.py` and are registered in
    `sources/__init__.py::REGISTRY`. They are constructed by `get_adapter()`
    with the kwargs documented in the module docstring above.
    """

    name: str

    def search(self, query: str, category_title: str) -> List[Paper]:
        """Run one search against this source.

        Args:
          query:          Boolean query string built by Stage 1
                          (e.g. `("LLM" OR "GPT") AND ("crisis" OR ...)`).
                          Some adapters simplify it before sending - that's OK.
          category_title: Human-readable category name. Stored on each
                          returned `Paper.search_category` so downstream
                          consumers can group results.

        Returns:
          List of `Paper` objects. MUST be empty on any failure (network,
          parse, throttle); MUST NOT raise. The pipeline depends on this.
        """
        ...
