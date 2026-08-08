"""The contract every source adapter implements.

    adapter = MySourceAdapter(user_agent=..., max_pages=..., max_results=...)
    papers = adapter.search(query, category_title)

Constructor kwargs are supplied by `get_adapter()` and are not part of the
protocol. Only `name` and `search()` are.
"""

from typing import Protocol

from ..schema import Paper


class SourceAdapter(Protocol):
    """A searchable paper repository.

    Implementations live in `sources/<source>.py` and are registered in
    `REGISTRY`.
    """

    name: str

    def search(self, query: str, category_title: str) -> list[Paper]:
        """Search this source for one category.

        Args:
          query: Boolean query from stage 1, such as
            `("LLM" OR "GPT") AND ("crisis" OR ...)`. Adapters may simplify
            it to suit their own query language.
          category_title: Category name, stored on each result's
            `search_category` so callers can group them.

        Returns:
          Matching papers, or an empty list on any failure. Never raises,
          the pipeline depends on that.
        """
        ...
