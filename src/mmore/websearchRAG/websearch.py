import logging
import os
import time
from typing import Dict, List, Literal

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException

from ..rag.llm import LLM, LLMConfig

logger = logging.getLogger(__name__)


class WebsearchOnly:
    """Class dedicated to performing web searches only
    Default provider: DuckDuckGo (free, no API key needed)
    Optional provider: Tavily (set TAVILY_API_KEY, pip install "mmore[rag,websearch]")
    """

    def __init__(
        self,
        region: str = "us-en",
        max_results: int = 10,
        provider: Literal["duckduckgo", "tavily"] = "duckduckgo",
        max_retries: int = 3,
    ):
        """Initialize the WebsearchOnly class with search parameters."""

        self.region = region
        self.max_results = max_results
        self.provider = provider
        self.max_retries = max_retries

        if provider == "tavily":
            try:
                from tavily import TavilyClient
            except ImportError:
                raise ImportError("Run: pip install mmore[rag,websearch]")
            api_key = os.getenv("TAVILY_API_KEY")

            if not api_key:
                raise ValueError("set TAVILY_API_KEY environment variable")

            self._tavily = TavilyClient(api_key=api_key)

    def _search_duckduckgo(self, query: str) -> List[Dict[str, str]]:
        """DDG search with exponential backoff retry - fixes the timeout error"""
        for attempt in range(self.max_retries):
            try:
                with DDGS() as ddgs:
                    results = list(
                        ddgs.text(
                            query, max_results=self.max_results, region=self.region
                        )
                    )
                return results

            except RatelimitException:
                wait = 2**attempt  # 1s -> 2s -> 4s

                logger.warning(
                    f"DDG rate limit hit, retrying in {wait}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                time.sleep(wait)

            except DDGSException as e:
                logger.error(f"DDG search error: {e}")
                return []

        logger.error("DDG search failed after all retries")
        return []

    def _search_tavily(self, query: str) -> List[Dict[str, str]]:
        """Tavily search : optional provider"""
        response = self._tavily.search(query, max_results=self.max_results)
        return [
            {
                "body": r.get("content", ""),
                "href": r.get("url", ""),
                "title": r.get("title", ""),
            }
            for r in response.get("results", [])
        ]

    def websearch_pipeline(self, query: str) -> List[Dict[str, str]]:
        """Perform a single web search."""

        if self.provider == "tavily":
            return self._search_tavily(query)
        return self._search_duckduckgo(query)

    def summarize_web_search(self, query: str, web_output: str) -> str:
        """Call LLM to summarize the current web output based on the original query, return a summary of the web search and the source."""
        llm = LLM.from_config(
            LLMConfig(llm_name="OpenMeditron/meditron3-8b", max_new_tokens=1200)
        )
        prompt = (
            f"Original Query: '{query}'\n"
            f"Web content: '{web_output}'\n"
            "Based on the original query and the web content, can you provide a response to the original query?"
        )
        response = llm.invoke(prompt).content
        assert isinstance(response, str)
        return response.strip()
