from typing import Any, Literal
from unittest.mock import MagicMock, patch

import pytest

from mmore.websearchRAG.config import WebsearchConfig
from mmore.websearchRAG.pipeline import WebsearchPipeline, extract_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pipeline(
    max_context_tokens=100,
    n_subqueries=2,
    n_loops=1,
    use_summary=False,
    use_rag=False,
    search_provider: Literal["duckduckgo", "tavily", "keenable"] = "tavily",
    subqueries=("sub1",),
    **config_overrides,
) -> Any:
    """Build a test WebsearchPipeline with a mocked LLM.

    When subqueries is None, generate_subqueries falls back to the mocked LLM.
    """
    config = WebsearchConfig(
        rag_config_path="dummy.yaml",
        output_file="dummy_out.json",
        max_context_tokens=max_context_tokens,
        n_subqueries=n_subqueries,
        n_loops=n_loops,
        use_summary=use_summary,
        use_rag=use_rag,
        search_provider=search_provider,
        **config_overrides,
    )
    with patch.object(WebsearchPipeline, "__init__", lambda self, cfg: None):
        pipeline = WebsearchPipeline(config)

    pipeline.config = config
    pipeline.rag_results = None
    pipeline._tokenizer = None
    pipeline._warned_fallback_tokenizer = False

    mock_llm = MagicMock()
    mock_llm.get_num_tokens = lambda text: len(text.split())  # 1 word = 1 token
    mock_llm.invoke.return_value = MagicMock(
        content="short answer: ok\ndetailed answer: detailed ok"
    )
    pipeline.llm = mock_llm

    pipeline.searcher = MagicMock()
    pipeline.searcher.websearch_pipeline.return_value = []

    if subqueries is not None:
        subs = list(subqueries)
        pipeline.generate_subqueries = lambda *a, **kw: subs

    return pipeline


def make_search_result(url, snippet, title="t"):
    """Shorthand for a web-search result dict (websearch.py output format)."""
    return {"body": snippet, "href": url, "title": title}


# ---------------------------------------------------------------------------
# Unit tests: extract_response
# ---------------------------------------------------------------------------


class TestExtractResponse:
    def test_string_input(self):
        assert extract_response("hello") == "hello"

    def test_list_of_strings(self):
        assert extract_response(["first", "second", "third"]) == "third"

    def test_list_of_dicts(self):
        assert extract_response([{"content": "from dict"}]) == "from dict"

    def test_list_of_dicts_missing_content(self):
        assert extract_response([{"other": "value"}]) == ""


# ---------------------------------------------------------------------------
# Unit tests: _clean_llm_output
# ---------------------------------------------------------------------------


class TestCleanLlmOutput:
    def test_strips_hf_special_tokens(self):
        p = make_pipeline()
        raw = "garbage<|eot_id|><|start_header_id|>assistant<|end_header_id|>actual answer"
        assert p._clean_llm_output(raw) == "actual answer"

    def test_returns_unchanged_without_delimiter(self):
        p = make_pipeline()
        assert p._clean_llm_output("normal text") == "normal text"


# ---------------------------------------------------------------------------
# Unit tests: _count_tokens, _truncate_to_token_limit, _fit_to_budget
# ---------------------------------------------------------------------------


class TestTokenHelpers:
    def test_count_tokens_delegates_to_llm_without_tokenizer(self):
        p = make_pipeline()
        assert p._count_tokens("one two three") == 3

    def test_count_tokens_uses_local_tokenizer_when_available(self):
        p = make_pipeline()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        p._tokenizer = mock_tokenizer

        assert p._count_tokens("some text") == 5
        mock_tokenizer.encode.assert_called_once_with(
            "some text", add_special_tokens=False
        )

    def test_truncate_no_op_when_within_limit(self):
        p = make_pipeline()
        assert (
            p._truncate_to_token_limit("one two three", max_tokens=10)
            == "one two three"
        )

    def test_truncate_shortens_text(self):
        p = make_pipeline()
        long_text = "word " * 100
        result = p._truncate_to_token_limit(long_text, max_tokens=5)

        assert len(result) < len(long_text)

    def test_truncate_with_local_tokenizer_slices_ids(self):
        p = make_pipeline()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(20))
        mock_tokenizer.decode.return_value = "truncated text"
        p._tokenizer = mock_tokenizer

        result = p._truncate_to_token_limit("some long text", max_tokens=5)

        assert result == "truncated text"
        mock_tokenizer.decode.assert_called_once_with(
            list(range(5)), skip_special_tokens=True
        )

    def test_fast_tokenizer_counts_and_truncates(self):
        p = make_pipeline(fast_tokenizer=True)
        # 12 chars -> 12 // 4 = 3 tokens
        assert p._count_tokens("twelve chars") == 3
        # Truncate to 2 tokens = 8 chars
        assert p._truncate_to_token_limit("twelve chars", max_tokens=2) == "twelve c"
        # No truncating when within limit
        assert (
            p._truncate_to_token_limit("twelve chars", max_tokens=5) == "twelve chars"
        )

    def test_fit_to_budget_truncates_content(self):
        # "system prompt" = 2 tokens, "prefix" = 1 token -> available = 20 - 3 = 17
        # The 10% safety margin means the result may be slightly shorter than 17.
        string = "word " * 30
        p = make_pipeline(max_context_tokens=20)
        result = p._fit_to_budget(string, "system prompt", "prefix")
        assert p._count_tokens(result) <= 17

    def test_fit_to_budget_raises_when_fixed_exceeds_max(self):
        p = make_pipeline(max_context_tokens=5)
        with pytest.raises(ValueError, match="exceed max_context_tokens"):
            p._fit_to_budget(
                "content", "this is a very long system prompt that exceeds everything"
            )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestSmoke:
    def test_process_record_returns_expected_keys(self):
        """No web results yields valid structure with empty sources."""
        p = make_pipeline(n_loops=1, n_subqueries=1, subqueries=None)

        result = p.process_record({"input": "What's the weather like today?"})

        assert result["query"] == "What's the weather like today?"
        for key in ("query", "short_answer", "detailed_answer", "sources"):
            assert key in result
        assert result["sources"] == {}

    def test_empty_query(self):
        p = make_pipeline(n_loops=1, subqueries=None)

        result = p.process_record({"input": ""})

        assert result["query"] == ""


# ---------------------------------------------------------------------------
# Snippet budget
# ---------------------------------------------------------------------------


class TestSnippetBudget:
    """Token-aware snippet accumulation and early stopping."""

    def test_all_snippets_collected_when_within_budget(self):
        p = make_pipeline(max_context_tokens=5000)

        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "small snippet one"),
            make_search_result("http://b.com", "small snippet two"),
        ]

        result = p.process_record({"input": "test query"})

        assert "http://a.com" in result["sources"]
        assert "http://b.com" in result["sources"]

    def test_budget_exhaustion_stops_accumulation(self):
        """First snippet (3 tokens) fits in 5-token budget; second (8 tokens) doesn't."""
        p = make_pipeline()
        p._compute_content_budget = lambda *_: 5

        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "alpha bravo charlie"),
            make_search_result(
                "http://b.com", "delta echo foxtrot golf hotel india juliet kilo"
            ),
        ]

        result = p.process_record({"input": "test query"})

        assert "http://a.com" in result["sources"]
        assert "http://b.com" not in result["sources"]

    def test_budget_exhaustion_skips_remaining_subqueries(self):
        """Once budget is exhausted, subsequent subqueries never call web_search."""
        p = make_pipeline(n_subqueries=3, subqueries=["sub1", "sub2", "sub3"])
        p._compute_content_budget = lambda *_: 5

        call_count = 0

        def counting_web_search(query):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Budget is exhausted after the first result
                return [
                    {
                        "url": f"http://{call_count}.com",
                        "snippet": "word " * 10,
                        "title": "t",
                    }
                ]
            return [
                {"url": f"http://{call_count}.com", "snippet": "other", "title": "t"}
            ]

        p.web_search = counting_web_search
        p.process_record({"input": "test query"})

        assert call_count == 1

    def test_snippet_at_exact_boundary_is_accepted(self):
        """When total + snippet == budget, snippet is accepted."""
        p = make_pipeline()

        # Make every text 10 tokens long
        p._count_tokens = lambda _: 10
        # Set content budget manually to 20 tokens
        p._compute_content_budget = lambda *_: 20

        p.searcher.websearch_pipeline.return_value = [
            make_search_result(
                "http://a.com", "first"
            ),  # 10 tokens, total=10, 10 > 20? No
            make_search_result(
                "http://b.com", "second"
            ),  # 10 tokens, total=20, 20 > 20? No
            make_search_result(
                "http://c.com", "third"
            ),  # 10 tokens, total=30, 30 > 20? Yes
        ]

        result = p.process_record({"input": "test query"})

        assert "http://a.com" in result["sources"]
        assert "http://b.com" in result["sources"]
        assert "http://c.com" not in result["sources"]

    def test_tiny_budget_raises(self):
        """With a budget too small for the fixed prompt parts, fail loudly."""
        p = make_pipeline(max_context_tokens=1)

        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "data"),
        ]

        with pytest.raises(ValueError, match="exceed max_context_tokens"):
            p.process_record({"input": "test query"})


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """(url, snippet) deduplication logic."""

    def test_exact_duplicate_is_skipped(self):
        p = make_pipeline()

        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "same snippet"),
            make_search_result("http://a.com", "same snippet"),
        ]

        result = p.process_record({"input": "test query"})

        assert len(result["sources"]["http://a.com"]) == 1

    def test_same_url_different_snippet_kept(self):
        p = make_pipeline()

        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "snippet alpha", title="Title A"),
            make_search_result("http://a.com", "snippet beta", title="Title B"),
        ]

        result = p.process_record({"input": "test query"})

        assert len(result["sources"]["http://a.com"]) == 2

    def test_same_snippet_different_url_kept(self):
        p = make_pipeline()

        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "identical text"),
            make_search_result("http://b.com", "identical text"),
        ]

        result = p.process_record({"input": "test query"})

        assert "http://a.com" in result["sources"]
        assert "http://b.com" in result["sources"]

    def test_dedup_persists_across_subqueries(self):
        p = make_pipeline(
            max_context_tokens=5000, n_subqueries=2, subqueries=["sub1", "sub2"]
        )

        call_count = 0

        def web_search_returning_same(query):
            nonlocal call_count
            call_count += 1
            return [
                {"url": "http://shared.com", "snippet": "shared content", "title": "t"}
            ]

        p.web_search = web_search_returning_same

        result = p.process_record({"input": "test query"})

        assert call_count == 2  # both subqueries searched
        assert len(result["sources"]["http://shared.com"]) == 1  # but only one kept

    def test_duplicates_do_not_consume_budget(self):
        """If a dup consumed budget, the third snippet would be rejected."""
        p = make_pipeline(max_context_tokens=5000)
        p._count_tokens = lambda _: 10
        # Budget fits 2 snippets (20 tokens) but not 3 (30 tokens)
        p._compute_content_budget = lambda *_: 25

        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", "real content"),  # 10 tokens, accepted
            make_search_result("http://a.com", "real content"),  # dup, skipped
            make_search_result(
                "http://b.com", "different content"
            ),  # 10 tokens, fits only if dup didn't count
        ]

        result = p.process_record({"input": "test query"})

        assert "http://b.com" in result["sources"]

    def test_dedup_persists_across_loops(self):
        """Same (url, snippet) in loop 1 is skipped because seen_results carries over."""
        p = make_pipeline(max_context_tokens=5000, n_loops=2)
        p.evaluate_subquery_relevance = MagicMock(return_value=True)

        call_count = 0

        def web_search_per_loop(query):
            nonlocal call_count
            call_count += 1
            # Same (url, snippet) but different title each loop
            return [
                {
                    "url": "http://a.com",
                    "snippet": "same snippet",
                    "title": f"Title Loop {call_count}",
                }
            ]

        p.web_search = web_search_per_loop

        result = p.process_record({"input": "test query"})

        # Only the first loop title is kept (second loop is a duplicate even if different title)
        assert result["sources"]["http://a.com"] == ["Title Loop 1"]


# ---------------------------------------------------------------------------
# Multi-loop budget and RAG context growth
# ---------------------------------------------------------------------------


class TestMultiLoopBudget:
    def test_second_loop_runs_when_relevant(self):
        p = make_pipeline(max_context_tokens=5000, n_loops=2)

        call_count = 0

        def counting_web_search(query):
            nonlocal call_count
            call_count += 1
            return [
                {"url": f"http://{call_count}.com", "snippet": "info", "title": "t"}
            ]

        p.web_search = counting_web_search
        p.evaluate_subquery_relevance = MagicMock(return_value=True)

        p.process_record({"input": "test query"})

        assert call_count == 2

    def test_second_loop_skipped_when_irrelevant(self):
        p = make_pipeline(max_context_tokens=5000, n_loops=2)

        call_count = 0

        def counting_web_search(query):
            nonlocal call_count
            call_count += 1
            return [
                {"url": f"http://{call_count}.com", "snippet": "info", "title": "t"}
            ]

        p.web_search = counting_web_search
        p.evaluate_subquery_relevance = MagicMock(return_value=False)

        p.process_record({"input": "test query"})

        assert call_count == 1

    def test_rag_context_grows_across_loops(self):
        """Loop 1's rag_for_llm should contain loop 0's detailed answer."""
        p = make_pipeline(max_context_tokens=5000, n_loops=2)

        rag_docs_seen = []

        def tracking_integrate(original, rag_doc, web_content):
            rag_docs_seen.append(rag_doc)
            return {"short": "s", "detailed": "long detailed answer for growth"}

        p.integrate_with_llm = tracking_integrate
        p.evaluate_subquery_relevance = MagicMock(return_value=True)
        p.web_search = lambda query: [
            {"url": "http://x.com", "snippet": "data", "title": "t"}
        ]

        p.process_record({"input": "test query"})

        assert rag_docs_seen[0] == ""
        assert "Prior answer:" in rag_docs_seen[1]
        assert "long detailed answer for growth" in rag_docs_seen[1]

    def test_snippet_budget_shrinks_with_growing_context(self):
        """Growing rag_for_llm increases synthesis prefix, reducing snippet budget."""
        p = make_pipeline(max_context_tokens=200, n_loops=2)

        budgets_seen = []
        original_compute = p._compute_content_budget

        def tracking_compute(*fixed_parts):
            budget = original_compute(*fixed_parts)
            budgets_seen.append(budget)
            return budget

        p._compute_content_budget = tracking_compute
        p.evaluate_subquery_relevance = MagicMock(return_value=True)
        p.web_search = lambda query: [
            {"url": "http://x.com", "snippet": "data", "title": "t"}
        ]

        p.llm.invoke.return_value = MagicMock(
            content="short answer: s\ndetailed answer: " + "word " * 30
        )

        p.process_record({"input": "test query"})

        # Ensure budget gets smaller across loops
        assert len(budgets_seen) >= 4
        assert budgets_seen[2] < budgets_seen[0]


# ---------------------------------------------------------------------------
# Summary budget (per-subquery)
# ---------------------------------------------------------------------------


class TestSummaryBudget:
    def test_large_snippet_excluded_by_summary_budget(self):
        """Per-subquery summary_budget excludes snippets that overflow it,
        even if the global snippet_budget has room."""
        p = make_pipeline()
        p._compute_content_budget = lambda *_: 10

        summary_inputs = []

        def tracking_summary(content, query):
            summary_inputs.append(content)
            return "summary"

        p.generate_summary = tracking_summary

        small = "word " * 3
        large = "word " * 50
        p.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", small),
            make_search_result("http://b.com", small),
            make_search_result("http://c.com", large),
        ]

        p.process_record({"input": "test query"})

        # First generate_summary call is the per-subquery one
        assert len(summary_inputs) >= 1
        per_subquery_input = summary_inputs[0]
        assert small.strip() in per_subquery_input
        assert large.strip() not in per_subquery_input

    def test_use_summary_bypasses_synthesis_overhead(self):
        """use_summary=True skips synthesis overhead, so a tight budget accepts
        what use_summary=False rejects."""
        snippet = "this snippet has six words total"

        p_no = make_pipeline(max_context_tokens=60, use_summary=False)
        p_no._compute_content_budget = MagicMock(return_value=3)
        p_no.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", snippet),
        ]
        result_no = p_no.process_record({"input": "test query"})

        p_yes = make_pipeline(max_context_tokens=60, use_summary=True)
        p_yes._compute_content_budget = MagicMock(return_value=60)
        p_yes.searcher.websearch_pipeline.return_value = [
            make_search_result("http://a.com", snippet),
        ]
        result_yes = p_yes.process_record({"input": "test query"})

        assert "http://a.com" not in result_no["sources"]
        assert "http://a.com" in result_yes["sources"]
