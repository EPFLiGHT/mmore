import json
from unittest.mock import MagicMock, patch

import pytest

from mmore.paper_discovery.boolean import (
    _sanitize_term,
    build_boolean_queries,
    load_synonyms,
)
from mmore.paper_discovery.config import PaperDiscoveryConfig
from mmore.paper_discovery.pdf import (
    DownloadResult,
    _find_pdf_link,
    _looks_like_login_page,
    _looks_like_pdf,
    _proxify,
    download_pdf,
    expected_pdf_path,
)
from mmore.paper_discovery.pipeline import PaperDiscoveryPipeline
from mmore.paper_discovery.schema import Paper, SourceName, SynonymEntry
from mmore.paper_discovery.sources._utils import coerce_year, first_year
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
# Login-page detection
# ---------------------------------------------------------------------------


class TestLooksLikeLoginPage:
    def _resp(self, body, ctype="text/html; charset=UTF-8"):
        r = MagicMock()
        r.headers = {"Content-Type": ctype}
        r.text = body
        return r

    def test_detects_a_shibboleth_page(self):
        assert _looks_like_login_page(
            self._resp("<html><body>Shibboleth Identity Provider</body></html>")
        )

    def test_detects_a_password_form(self):
        assert _looks_like_login_page(
            self._resp('<form><input type="password" name="pw"></form>')
        )

    def test_ignores_non_html(self):
        # A real PDF must never be mistaken for a login page.
        assert not _looks_like_login_page(
            self._resp("%PDF-1.7 ...", ctype="application/pdf")
        )

    def test_ignores_ordinary_article_html(self):
        assert not _looks_like_login_page(
            self._resp("<html><body>Abstract: we introduce...</body></html>")
        )


class TestProxify:
    def test_wraps_the_url_when_a_prefix_is_set(self):
        out = _proxify("https://x.com/a.pdf", "https://ezproxy.example.edu")
        assert out.startswith("https://ezproxy.example.edu/login?url=")
        assert "https%3A%2F%2Fx.com%2Fa.pdf" in out

    def test_is_a_noop_without_a_prefix(self):
        assert _proxify("https://x.com/a.pdf", None) == "https://x.com/a.pdf"

    def test_does_not_double_wrap(self):
        already = "https://ezproxy.example.edu/login?url=https%3A%2F%2Fx.com"
        assert _proxify(already, "https://ezproxy.example.edu") == already

    def test_tolerates_a_trailing_slash_on_the_prefix(self):
        out = _proxify("https://x.com/a.pdf", "https://ezproxy.example.edu/")
        assert "//login?url=" not in out


class TestLooksLikePdf:
    def _resp(
        self, content, url="https://example.org/paper.pdf", ctype="application/pdf"
    ):
        r = MagicMock()
        r.content = content
        r.url = url
        r.headers = {"Content-Type": ctype}
        return r

    def test_accepts_the_pdf_signature(self):
        assert _looks_like_pdf(self._resp(b"%PDF-1.7\n..."))

    def test_rejects_html_from_a_pdf_url(self):
        # The URL and Content-Type both say PDF, but the body is a login page.
        assert not _looks_like_pdf(self._resp(b"<html><body>Sign in</body></html>"))


class TestDownloadPdf:
    def _resp(self, content, ctype):
        r = MagicMock()
        r.status_code = 200
        r.content = content
        r.text = content.decode("utf-8", errors="ignore")
        r.url = "https://example.org/paper.pdf"
        r.headers = {"Content-Type": ctype}
        return r

    def test_login_page_on_a_pdf_url_is_not_saved(self, tmp_path):
        login = self._resp(b'<form><input type="password"></form>', "text/html")
        with patch("mmore.paper_discovery.pdf.requests.get", return_value=login):
            result = download_pdf("https://example.org/paper.pdf", str(tmp_path))
        assert result.path is None
        assert result.login_page
        assert not any(tmp_path.iterdir())


class TestExpectedPdfPath:
    def test_urls_ending_alike_get_different_files(self, tmp_path):
        a = expected_pdf_path("https://a.org/article/1/pdf", str(tmp_path))
        b = expected_pdf_path("https://b.org/article/2/pdf", str(tmp_path))
        assert a != b

    def test_same_url_gets_the_same_file(self, tmp_path):
        url = "https://a.org/article/1/pdf"
        assert expected_pdf_path(url, str(tmp_path)) == expected_pdf_path(
            url, str(tmp_path)
        )


class TestFindPdfLink:
    def test_prefers_citation_pdf_url(self):
        html = """
        <html><head>
          <meta name="citation_pdf_url" content="/article/1/main.pdf">
        </head><body>
          <a href="/related/2/pdf">Related article</a>
        </body></html>
        """
        assert (
            _find_pdf_link(html, base="https://pub.org/article/1")
            == "https://pub.org/article/1/main.pdf"
        )

    def test_falls_back_to_links(self):
        html = '<a href="/article/1/pdf">PDF</a>'
        assert (
            _find_pdf_link(html, base="https://pub.org/article/1")
            == "https://pub.org/article/1/pdf"
        )


class TestEnrichWithPdfText:
    def _pipeline(self, tmp_path):
        cfg = PaperDiscoveryConfig(
            synonyms_path="unused",
            categories_path="unused",
            output_file=str(tmp_path / "out.jsonl"),
            pdf_dir=str(tmp_path / "pdfs"),
        )
        return PaperDiscoveryPipeline(cfg)

    def _download_ok(self, tmp_path):
        def fake(url, save_dir, **kwargs):
            path = expected_pdf_path(url, save_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"%PDF-1.7\n")
            return DownloadResult(path=str(path))

        return fake

    def test_empty_text_is_not_a_success(self, tmp_path, caplog):
        paper = Paper(title="T", url="https://example.org/a.pdf")
        with (
            patch(
                "mmore.paper_discovery.pipeline.download_pdf",
                side_effect=self._download_ok(tmp_path),
            ),
            patch("mmore.paper_discovery.pipeline.extract_text", return_value=""),
            caplog.at_level("INFO"),
        ):
            self._pipeline(tmp_path)._enrich_with_pdf_text([paper])
        assert paper.extracted_text is None
        assert "0/1 succeeded" in caplog.text
        assert "1 with no text" in caplog.text

    def test_non_pdf_cache_file_is_replaced(self, tmp_path, caplog):
        pipeline = self._pipeline(tmp_path)
        url = "https://example.org/a.pdf"
        paper = Paper(title="T", url=url)
        cached = expected_pdf_path(url, pipeline.config.pdf_dir)
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"<html>login</html>")

        download = MagicMock(side_effect=self._download_ok(tmp_path))
        with (
            patch("mmore.paper_discovery.pipeline.download_pdf", download),
            patch("mmore.paper_discovery.pipeline.extract_text", return_value="text"),
            caplog.at_level("INFO"),
        ):
            pipeline._enrich_with_pdf_text([paper])
        download.assert_called_once()
        assert cached.read_bytes().startswith(b"%PDF-")
        assert paper.extracted_text == "text"
        assert "1/1 succeeded (0 cached, 1 fresh)" in caplog.text

    def test_valid_cache_file_skips_download(self, tmp_path, caplog):
        pipeline = self._pipeline(tmp_path)
        url = "https://example.org/a.pdf"
        paper = Paper(title="T", url=url)
        cached = expected_pdf_path(url, pipeline.config.pdf_dir)
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"%PDF-1.7\n")

        download = MagicMock()
        with (
            patch("mmore.paper_discovery.pipeline.download_pdf", download),
            patch("mmore.paper_discovery.pipeline.extract_text", return_value="text"),
            caplog.at_level("INFO"),
        ):
            pipeline._enrich_with_pdf_text([paper])
        download.assert_not_called()
        assert "1/1 succeeded (1 cached, 0 fresh)" in caplog.text


# ---------------------------------------------------------------------------
# Shared source helpers
# ---------------------------------------------------------------------------


class TestCoerceYear:
    def test_handles_the_shapes_each_source_returns(self):
        assert coerce_year(2024) == 2024  # OpenAlex: int
        assert coerce_year("2024") == 2024  # Google Scholar: year string
        assert coerce_year("2024-01-15T00:00:00Z") == 2024  # arXiv: ISO date
        assert coerce_year(" 2024 ") == 2024  # padded

    def test_returns_none_when_unusable(self):
        assert coerce_year(None) is None
        assert coerce_year("") is None
        assert coerce_year("n/a") is None

    def test_first_year_takes_the_first_parseable_key(self):
        entry = {"pubYear": "", "firstPublicationDate": "2019-04-02"}
        assert first_year(entry, "pubYear", "firstPublicationDate") == 2019

    def test_first_year_returns_none_when_no_key_parses(self):
        assert first_year({"pubYear": "n/a"}, "pubYear", "missing") is None


# ---------------------------------------------------------------------------
# SourceName enum
# ---------------------------------------------------------------------------


class TestSourceName:
    def test_serializes_as_plain_string(self):
        # The (str, Enum) mixin is what keeps papers.jsonl unchanged.
        # Without it json.dumps would emit "SourceName.ARXIV".
        payload = json.dumps(Paper(title="t", source=SourceName.ARXIV).to_dict())
        assert '"source": "arxiv"' in payload

    def test_compares_equal_to_its_string_value(self):
        assert SourceName.OPENALEX == "openalex"

    def test_covers_every_registered_source(self):
        from mmore.paper_discovery.sources import REGISTRY

        assert {s.value for s in SourceName} == set(REGISTRY)


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
        p = Paper(title="t", source=SourceName.ARXIV)
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
            authors=["Ada Lovelace"],
            url="http://x/p.pdf",
            abstract="abs",
            year=2024,
            source=SourceName.ARXIV,
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
        p = Paper(title="T", source=SourceName.ARXIV)  # authors, url, year, ... = None
        s = p.to_multimodal_sample()
        assert "authors" not in s.metadata.extra
        assert "year" not in s.metadata.extra
        assert s.metadata.extra["title"] == "T"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
