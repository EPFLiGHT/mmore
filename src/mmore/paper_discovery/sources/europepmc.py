import logging
import time
from typing import List, Optional

import requests

from ..schema import Paper
from .base import SourceAdapter

logger = logging.getLogger(__name__)

API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
RATE_LIMIT_SECONDS = 1.0


class EuropePmcAdapter(SourceAdapter):
    name = "europepmc"

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
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": min(25, self.max_results - len(papers)),
                "cursorMark": cursor,
            }
            try:
                r = requests.get(
                    API_URL, params=params, headers=self.headers, timeout=30
                )
                r.raise_for_status()
                data = r.json()
            except (requests.RequestException, ValueError) as e:
                logger.warning("Europe PMC request failed: %s", e)
                break

            for entry in data.get("resultList", {}).get("result", []):
                papers.append(self._to_paper(entry, category_title))
                if len(papers) >= self.max_results:
                    return papers

            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
            time.sleep(RATE_LIMIT_SECONDS)

        return papers

    def _to_paper(self, entry: dict, category_title: str) -> Paper:
        urls = entry.get("fullTextUrlList", {}).get("fullTextUrl", [])
        pdf_url = next(
            (u.get("url") for u in urls if u.get("docementStyle", "").lower() == "pdf"),
            None,
        )
        landing = next((u.get("url") for u in urls), None)
        year = _coerce_year(entry)

        return Paper(
            title=entry.get("title"),
            authors=entry.get("authorString"),
            url=pdf_url or landing,
            abstract=entry.get("abstractText"),
            year=year,
            source="europepmc",
            search_category=category_title,
        )


def _coerce_year(entry: dict) -> Optional[int]:
    for key in ("pubYear", "firstPublicationDate"):
        v = entry.get(key)
        if v:
            try:
                return int(str(v)[:4])
            except ValueError:
                continue
    return None
