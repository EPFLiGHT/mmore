import logging
import time
from typing import List, Optional

import requests

from ..schema import Paper, SourceName
from ._utils import first_year
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
            (u.get("url") for u in urls if u.get("documentStyle", "").lower() == "pdf"),
            None,
        )
        landing = next((u.get("url") for u in urls), None)
        year = first_year(entry, "pubYear", "firstPublicationDate")

        return Paper(
            title=entry.get("title"),
            authors=_parse_authors(entry),
            url=pdf_url or landing,
            abstract=entry.get("abstractText"),
            year=year,
            source=SourceName.EUROPEPMC,
            search_category=category_title,
        )


def _parse_authors(entry: dict) -> Optional[List[str]]:
    """Prefer the structured `authorList.author[].fullName` (available with
    `resultType=core`). Fall back to naively splitting the pre-joined
    `authorString` when the structured shape is missing - imperfect
    (some names contain commas) but better than dropping the field.
    """
    author_list = (entry.get("authorList") or {}).get("author") or []
    names = [a["fullName"] for a in author_list if a.get("fullName")]
    if names:
        return names
    raw = entry.get("authorString")
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or None
