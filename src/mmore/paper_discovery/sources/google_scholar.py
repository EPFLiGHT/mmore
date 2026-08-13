"""Google Scholar adapter. Opt-in and best-effort, since Scholar serves
captchas to automated callers.

Needs the optional `scholarly` package, imported lazily so the rest of the
module works without it.
"""

import logging

from ..schema import Paper, SourceName
from ._utils import coerce_year
from .base import SourceAdapter

logger = logging.getLogger(__name__)


class GoogleScholarAdapter(SourceAdapter):
    name = "google_scholar"

    def __init__(
        self,
        user_agent: str = "mmore-paper-discovery/1.0",
        max_pages: int = 1,
        max_results: int = 20,
    ):
        # `scholarly` drives its own HTTP transport and pagination, so
        # user_agent and max_pages are accepted for parity with the other
        # adapters (get_adapter passes the same kwargs to all of them) but
        # cannot be forwarded.
        del user_agent, max_pages
        self.max_results = max_results

    def search(self, query: str, category_title: str) -> list[Paper]:
        try:
            from scholarly import scholarly
        except ImportError:
            logger.warning(
                "scholarly not installed; install with `pip install scholarly` "
                "to enable Google Scholar source"
            )
            return []

        papers: list[Paper] = []
        try:
            results = scholarly.search_pubs(query)
            for _ in range(self.max_results):
                try:
                    item = next(results)
                except StopIteration:
                    break
                bib = item.get("bib", {})
                papers.append(
                    Paper(
                        title=bib.get("title"),
                        authors=list(bib.get("author") or []) or None,
                        url=item.get("pub_url") or item.get("eprint_url"),
                        abstract=bib.get("abstract"),
                        year=coerce_year(bib.get("pub_year")),
                        source=SourceName.GOOGLE_SCHOLAR,
                        search_category=category_title,
                    )
                )
        except Exception as e:
            logger.warning("Google Scholar request failed: %s", e)
            if not papers:
                logger.warning(
                    "Google Scholar returned nothing, possibly a captcha or throttling"
                )
        return papers
