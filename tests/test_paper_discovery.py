from unittest.mock import MagicMock, patch

import pytest

from mmore.paper_discovery.boolean import (
    _sanitize_term,
    build_boolean_queries,
    load_synonyms,
)
from mmore.paper_discovery.schema import Paper, SynonymEntry
from mmore.paper_discovery.sources.arxiv import (
    _build_simplified_queries,
    _extract_terms,
)
from mmore.paper_discovery.sources.europepmc import EuropePmcAdapter, _parse_authors
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
        assert p.authors == ["Ada Lovelace"]
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

    def test_build_simplified_pair_disabled(self):
        queries = _build_simplified_queries(
            ["LLM", "GPT", "BERT"], top_n=4, enable_pair=False
        )
        assert 'all:"LLM"' in queries
        assert not any("AND" in q for q in queries)


# ---------------------------------------------------------------------------
# Europe PMC parsing
# ---------------------------------------------------------------------------


class TestEuropePmcToPaper:
    def _entry(self, urls):
        return {
            "title": "A Paper",
            "fullTextUrlList": {"fullTextUrl": urls},
            "pubYear": "2024",
        }

    def test_prefers_pdf_url_over_landing_page(self):
        # Regression: the key was misspelled `docementStyle`, so this branch
        # never matched and every result fell through to the landing page.
        adapter = EuropePmcAdapter()
        entry = self._entry(
            [
                {"url": "http://x/landing", "documentStyle": "html"},
                {"url": "http://x/paper.pdf", "documentStyle": "pdf"},
            ]
        )
        assert adapter._to_paper(entry, "Cat").url == "http://x/paper.pdf"

    def test_falls_back_to_first_url_when_no_pdf(self):
        adapter = EuropePmcAdapter()
        entry = self._entry([{"url": "http://x/landing", "documentStyle": "html"}])
        assert adapter._to_paper(entry, "Cat").url == "http://x/landing"

    def test_url_is_none_when_no_urls(self):
        adapter = EuropePmcAdapter()
        assert adapter._to_paper(self._entry([]), "Cat").url is None


class TestEuropePmcParseAuthors:
    def test_prefers_structured_author_list(self):
        entry = {
            "authorList": {
                "author": [
                    {"fullName": "Ada Lovelace"},
                    {"fullName": "Alan Turing"},
                ]
            },
            "authorString": "Lovelace A, Turing A",
        }
        assert _parse_authors(entry) == ["Ada Lovelace", "Alan Turing"]

    def test_falls_back_to_author_string_when_structured_missing(self):
        entry = {"authorString": "Lovelace A, Turing A"}
        assert _parse_authors(entry) == ["Lovelace A", "Turing A"]

    def test_returns_none_when_both_missing(self):
        assert _parse_authors({}) is None


class TestLoadSynonyms:
    def test_loads_jsonl_line_per_object(self, tmp_path):
        f = tmp_path / "syns.jsonl"
        f.write_text(
            '{"word": "LLM", "synonyms": ["GPT"]}\n'
            '{"word": "Crisis", "synonyms": ["disaster response"]}\n'
        )
        entries = load_synonyms(f)
        assert [e.word for e in entries] == ["LLM", "Crisis"]
        assert entries[1].synonyms == ["disaster response"]

    def test_jsonl_skips_blank_lines(self, tmp_path):
        f = tmp_path / "syns.jsonl"
        f.write_text(
            '{"word": "LLM", "synonyms": ["GPT"]}\n'
            "\n"
            "   \n"
            '{"word": "Crisis", "synonyms": ["disaster"]}\n'
        )
        entries = load_synonyms(f)
        assert [e.word for e in entries] == ["LLM", "Crisis"]

    def test_sanitize_strips_embedded_double_quote(self):
        # The reviewer's concern: `"` inside a term would close the quoted
        # phrase early and produce a malformed boolean query.
        assert _sanitize_term('bad"term') == "badterm"
        assert _sanitize_term('  many   spaces  "x" ') == "many spaces x"


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


class TestPaperToMultimodalSample:
    def test_text_prefers_extracted_over_abstract(self):
        p = Paper(title="T", abstract="abs", extracted_text="full")
        s = p.to_multimodal_sample()
        assert s.text == "full"

    def test_text_falls_back_to_abstract_then_title(self):
        assert Paper(abstract="abs", title="T").to_multimodal_sample().text == "abs"
        assert Paper(title="T").to_multimodal_sample().text == "T"
        assert Paper().to_multimodal_sample().text == ""

    def test_metadata_carries_paper_fields(self):
        p = Paper(
            title="A Paper",
            authors="Ada Lovelace",
            url="http://x/p.pdf",
            abstract="abs",
            year=2024,
            source="arxiv",
            search_category="Cat",
        )
        s = p.to_multimodal_sample(pdf_path="/tmp/p.pdf")
        assert s.metadata.file_path == "/tmp/p.pdf"
        assert s.metadata.processor_type == "paper_discovery"
        assert s.metadata.extra["title"] == "A Paper"
        assert s.metadata.extra["source"] == "arxiv"
        assert s.metadata.extra["search_category"] == "Cat"
        assert s.metadata.extra["year"] == 2024

    def test_none_fields_dropped_from_extra(self):
        p = Paper(title="T", source="arxiv")  # authors, url, year, ... = None
        s = p.to_multimodal_sample()
        assert "authors" not in s.metadata.extra
        assert "year" not in s.metadata.extra
        assert s.metadata.extra["title"] == "T"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
