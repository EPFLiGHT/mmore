"""arXiv adapter.

arXiv's query language is term-based — it chokes on rich boolean expressions —
so we simplify the input query into a small set of `all:"<term>"` queries.

Strict rate limit: 1 request / 3 seconds (arXiv ToS). Do NOT lower this.
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import requests

from ..schema import Paper
from .base import SourceAdapter

logger = logging.getLogger(__name__)

API_URL = "http://export.arxiv.org/api/query"
RATE_LIMIT_SECONDS = 3.0  # arXiv ToS — do not lower.
BACKOFF_SECONDS = 30.0  # Cooldown after a 429 — arXiv recovers quickly with space.
REQUEST_TIMEOUT = 60  # arXiv cold queries often take 30-45s.
NS = {"atom": "http://www.w3.org/2005/Atom"}

# Words to drop when extracting key terms from a boolean string.
STOPWORDS = {"or", "and", "the", "for", "with", "data", "of", "in", "on", "to"}


class ArxivAdapter(SourceAdapter):
    name = "arxiv"

    def __init__(
        self,
        user_agent: str = "mmore-paper-discovery/1.0",
        max_pages: int = 2,
        max_results: int = 50,
        category_map: Optional[Dict[str, str]] = None,
    ):
        self.headers = {"User-Agent": user_agent}
        self.max_pages = max_pages
        self.max_results = max_results
        self.category_map = category_map or {}

    def search(self, query: str, category_title: str) -> List[Paper]:
        terms = _extract_terms(query)
        if not terms:
            logger.info("arXiv: no usable terms from query, skipping")
            return []

        simplified = _build_simplified_queries(terms, top_n=4)
        cat_code = self._cat_code_for(category_title)
        if cat_code:
            simplified.append(f"cat:{cat_code}")

        papers: List[Paper] = []
        for q in simplified:
            for page in range(self.max_pages):
                start = page * 25
                params = {
                    "search_query": q,
                    "start": start,
                    "max_results": 25,
                }
                try:
                    r = requests.get(
                        API_URL,
                        params=params,
                        headers=self.headers,
                        timeout=REQUEST_TIMEOUT,
                    )
                    r.raise_for_status()
                except requests.HTTPError as e:
                    status = getattr(e.response, "status_code", None)
                    if status == 429:
                        logger.warning("arXiv 429 - backing off %ss", BACKOFF_SECONDS)
                        time.sleep(BACKOFF_SECONDS)
                    else:
                        logger.warning("arXiv request failed (%s): %s", q, e)
                        time.sleep(RATE_LIMIT_SECONDS)
                    break
                except requests.RequestException as e:
                    logger.warning("arXiv request failed (%s): %s", q, e)
                    time.sleep(RATE_LIMIT_SECONDS)
                    break

                entries = _parse_atom(r.text, category_title)
                if not entries:
                    time.sleep(RATE_LIMIT_SECONDS)
                    break
                papers.extend(entries)
                time.sleep(RATE_LIMIT_SECONDS)

                if len(papers) >= self.max_results:
                    return papers[: self.max_results]
        return papers

    def _cat_code_for(self, category_title: str) -> Optional[str]:
        title_lower = category_title.lower()
        for needle, code in self.category_map.items():
            if needle.lower() in title_lower:
                return code
        return None


def _extract_terms(boolean_query: str) -> List[str]:
    """Pull double-quoted phrases out of the boolean string."""
    quoted = re.findall(r'"([^"]+)"', boolean_query)
    return [t for t in quoted if t.lower() not in STOPWORDS]


def _build_simplified_queries(terms: List[str], top_n: int = 4) -> List[str]:
    chosen = terms[:top_n]
    queries = [f'all:"{t}"' for t in chosen]
    if len(chosen) >= 2:
        queries.append(f'all:"{chosen[0]}" AND all:"{chosen[1]}"')
    return queries


def _parse_atom(xml_text: str, category_title: str) -> List[Paper]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning("arXiv parse failed: %s", e)
        return []

    out: List[Paper] = []
    for entry in root.findall("atom:entry", NS):
        title = _text(entry, "atom:title")
        summary = _text(entry, "atom:summary")
        pub = _text(entry, "atom:published")
        year = _coerce_year(pub)
        authors = ", ".join(
            _text(a, "atom:name") or "" for a in entry.findall("atom:author", NS)
        ).strip(", ")
        pdf_url = None
        for link in entry.findall("atom:link", NS):
            if link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
                break
        landing = _text(entry, "atom:id")

        out.append(
            Paper(
                title=(title or "").strip() or None,
                authors=authors or None,
                url=pdf_url or landing,
                abstract=(summary or "").strip() or None,
                year=year,
                source="arxiv",
                search_category=category_title,
            )
        )
    return out


def _text(elem, path: str) -> Optional[str]:
    node = elem.find(path, NS)
    return node.text if node is not None else None


def _coerce_year(date_str: Optional[str]) -> Optional[int]:
    if not date_str or len(date_str) < 4:
        return None
    try:
        return int(date_str[:4])
    except ValueError:
        return None
