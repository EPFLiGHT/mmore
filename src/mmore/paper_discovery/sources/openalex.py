import logging
import time
from typing import Dict, List, Optional

import requests

from ..schema import Paper
from .base import SourceAdapter

logger = logging.getLogger(__name__)

API_URL = "https://api.openalex.org/works"
RATE_LIMIT_SECONDS = 1.0


class OpenAlexAdapter(SourceAdapter):
    name = "openalex"

    def __init__(
        self,
        user_agent: str = "mmore-paper-discovery/1.0",
        max_pages: int = 3,
        max_results: int = 50,
    ):
        self.headers = {"User-Agent": user_agent}
        self.max_pages = max_pages
        self.max_results = max_results

    def search(self, query: str, category_title: str) -> List[Paper]:
        papers: List[Paper] = []
        cursor = "*"

        for _ in range(self.max_pages):
            params = {
                "search": query,
                "per-page": min(25, self.max_results - len(papers)),
                "cursor": cursor,
            }
            try:
                r = requests.get(
                    API_URL, params=params, headers=self.headers, timeout=30
                )
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                logger.warning("OpenAlex request failed: %s", e)
                break

            for work in data.get("results", []):
                papers.append(self._to_paper(work, category_title))
                if len(papers) >= self.max_results:
                    return papers

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
            time.sleep(RATE_LIMIT_SECONDS)

        return papers

    def _to_paper(self, work: dict, category_title: str) -> Paper:
        authors = ", ".join(
            a.get("author", {}).get("display_name", "")
            for a in work.get("authorships", [])
            if a.get("author", {}).get("display_name")
        )
        loc = work.get("primary_location") or {}
        pdf_url = loc.get("pdf_url")
        landing = loc.get("landing_page_url") or work.get("id")

        return Paper(
            title=work.get("title"),
            authors=authors or None,
            url=pdf_url or landing,
            abstract=_rebuild_abstract(work.get("abstract_inverted_index")),
            year=_coerce_int(work.get("publication_year")),
            source="openalex",
            search_category=category_title,
        )


def _rebuild_abstract(inverted: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """OpenAlex returns abstracts as {token: [positions]} not a string."""
    if not inverted:
        return None
    pairs = [(pos, tok) for tok, positions in inverted.items() for pos in positions]
    pairs.sort()
    return " ".join(tok for _, tok in pairs)


def _coerce_int(value) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
