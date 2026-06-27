"""Google Scholar adapter — opt-in, best-effort. Captcha-prone.

Requires the optional `scholarly` package. We import lazily so that
the rest of the module works without it.
"""

import logging
from typing import List

from ..schema import Paper
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
        self.max_results = max_results

    def search(self, query: str, category_title: str) -> List[Paper]:
        try:
            from scholarly import scholarly
        except ImportError:
            logger.warning(
                "scholarly not installed; install with `pip install scholarly` "
                "to enable Google Scholar source"
            )
            return []

        papers: List[Paper] = []
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
                        authors=", ".join(bib.get("author", []) or []) or None,
                        url=item.get("pub_url") or item.get("eprint_url"),
                        abstract=bib.get("abstract"),
                        year=_safe_year(bib.get("pub_year")),
                        source="google_scholar",
                        search_category=category_title,
                    )
                )
        except Exception as e:
            logger.warning("Google Scholar request failed: %s", e)
            if not papers:
                logger.warning(
                    "Empty Google Scholar yield — possible captcha/throttling"
                )
        return papers


def _safe_year(value):
    try:
        return int(value) if value else None
    except (TypeError, ValueError):
        return None
