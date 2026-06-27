from unittest.mock import MagicMock, patch

import pytest

from mmore.paper_discovery.boolean import build_boolean_queries
from mmore.paper_discovery.schema import Paper, SynonymEntry
from mmore.paper_discovery.sources.arxiv import (
    _build_simplified_queries,
    _extract_terms,
)
from mmore.paper_discovery.sources.openalex import OpenAlexAdapter, _rebuild_abstract

# ---------------------------------------------------------------------------
# Stage 1: boolean builder
# ---------------------------------------------------------------------------


class TestBooleanBuilder:
    def test_or_group_is_alphabetical(self):
        syns = [SynonymEntry(word="LLM", synonyms=["GPT", "foundation model"])]
        queries = build_boolean_queries(syns, {"Cat": ["LLM"]})
        assert len(queries) == 1
        # OR group is sorted, AND-joined
        assert '"GPT"' in queries[0].boolean_combination
        assert '"LLM"' in queries[0].boolean_combination

    def test_unknown_word_logged_not_raised(self, caplog):
        syns = [SynonymEntry(word="LLM", synonyms=["GPT"])]
        queries = build_boolean_queries(syns, {"Cat": ["LLM", "UnknownWord"]})
        assert len(queries) == 1
        assert "UnknownWord" in caplog.text

    def test_empty_category_dropped(self):
        syns = [SynonymEntry(word="LLM", synonyms=["GPT"])]
        queries = build_boolean_queries(syns, {"Cat": ["NotPresent"]})
        assert queries == []


# ---------------------------------------------------------------------------
# OpenAlex helpers + adapter
# ---------------------------------------------------------------------------


class TestRebuildAbstract:
    def test_empty(self):
        assert _rebuild_abstract(None) is None
        assert _rebuild_abstract({}) is None

    def test_orders_tokens_by_position(self):
        inverted = {"world": [1], "hello": [0]}
        assert _rebuild_abstract(inverted) == "hello world"


class TestOpenAlexAdapter:
    def test_returns_empty_on_request_failure(self):
        adapter = OpenAlexAdapter(max_pages=1, max_results=10)
        with patch(
            "mmore.paper_discovery.sources.openalex.requests.get",
            side_effect=Exception("boom"),
        ):
            assert adapter.search("anything", "Cat") == []

    def test_parses_one_result(self):
        adapter = OpenAlexAdapter(max_pages=1, max_results=5)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "A Paper",
                    "publication_year": 2024,
                    "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
                    "primary_location": {"pdf_url": "http://x/p.pdf"},
                    "abstract_inverted_index": {"hi": [0]},
                }
            ],
            "meta": {"next_cursor": None},
        }
        mock_response.raise_for_status = MagicMock()
        with patch(
            "mmore.paper_discovery.sources.openalex.requests.get",
            return_value=mock_response,
        ):
            papers = adapter.search("q", "MyCat")
        assert len(papers) == 1
        p = papers[0]
        assert isinstance(p, Paper)
        assert p.title == "A Paper"
        assert p.year == 2024
        assert p.authors == "Ada Lovelace"
        assert p.url == "http://x/p.pdf"
        assert p.abstract == "hi"
        assert p.source == "openalex"
        assert p.search_category == "MyCat"


# ---------------------------------------------------------------------------
# arXiv simplification
# ---------------------------------------------------------------------------


class TestArxivSimplification:
    def test_extract_terms_drops_stopwords(self):
        q = '"LLM" OR "GPT" AND "data"'
        assert _extract_terms(q) == ["LLM", "GPT"]

    def test_extract_terms_empty(self):
        assert _extract_terms("") == []

    def test_build_simplified_includes_pair(self):
        queries = _build_simplified_queries(["LLM", "GPT", "BERT"], top_n=4)
        assert 'all:"LLM"' in queries
        assert 'all:"GPT"' in queries
        assert any("AND" in q for q in queries)


# ---------------------------------------------------------------------------
# Smoke test on Paper schema
# ---------------------------------------------------------------------------


class TestPaperSchema:
    def test_to_dict_includes_all_fields(self):
        p = Paper(title="t", source="arxiv")
        d = p.to_dict()
        for k in (
            "title",
            "authors",
            "url",
            "abstract",
            "year",
            "extracted_text",
            "source",
            "search_category",
        ):
            assert k in d

    def test_nullable_fields_default_none(self):
        p = Paper()
        assert p.title is None
        assert p.year is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
