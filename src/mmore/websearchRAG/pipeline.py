import json
import math
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from rich.markup import escape

from ..rag.llm import LLM, LLMConfig
from ..run_rag import rag
from ..ux import Color, model_loading_seconds, progress, step_intro, step_summary
from .config import WebsearchConfig
from .logging_config import WEBSEARCH_EMOJI, logger
from .websearch import WebsearchOnly

# --- Prompt constants ---

SUMMARY_SYSTEM_MSG = (
    "You are an extractive summarizer. Use only the provided context, no external knowledge. "
    "Keep the summary concise and factual."
)
SUMMARY_PREFIX = "Question: {query}\n\n---CONTEXT---\n"
SUMMARY_SUFFIX = (
    "\n---END CONTEXT---\n\n"
    "Extract and summarize only the information relevant to the question above.\n"
    "If the context contains no useful information, respond exactly with: 'NO_USEFUL_INFORMATION'"
)

RELEVANCE_SYSTEM_MSG = "You are a binary classifier. You must respond with exactly one word: 'yes' or 'no'."
RELEVANCE_PROMPT = (
    "Original query:\n{query}\n\n"
    "Previous subqueries that contribute to understanding:\n{previous_subqueries}\n\n"
    "New subqueries:\n{current_subqueries}\n\n"
    "Are any of the new subqueries relevant in the context of the original query and previous subqueries? "
    "Respond with a single word: 'yes' or 'no'."
)

SUBQUERY_SYSTEM_MSG = "You are a search query generator. Output only the requested subqueries in the specified format."
SUBQUERY_TASK = (
    "Generate exactly {n} independent web-search subqueries that together cover the question comprehensively.\n"
    "Each subquery must be concise (≤30 words) and search-engine friendly.\n\n"
    "Output format (one per line, no extra text):\n"
    "subquery <i>: <query>\n"
)
SUBQUERY_TASK_WITH_CONTEXT = (
    "Partial answer so far:\n{current_context}\n\n"
    "Generate exactly {n} independent web-search subqueries to fill gaps in the partial answer.\n"
    "Each subquery must be concise (≤30 words) and search-engine friendly.\n"
    "Do not repeat aspects already covered by the partial answer.\n\n"
    "Output format (one per line, no extra text):\n"
    "subquery <i>: <query>\n"
)

SYNTHESIS_SYSTEM_MSG = (
    "You are a research assistant. Synthesize the provided sources into a clear answer. "
    "Do not introduce information beyond what is given."
)
SYNTHESIS_PREFIX = "Question: {original}\n\n---RAG SOURCES---\n{rag_doc}\n---END RAG SOURCES---\n\n---WEB SOURCES---\n"
SYNTHESIS_SUFFIX = (
    "\n---END WEB SOURCES---\n\n"
    "Respond in exactly this format (keep the labels):\n"
    "short answer: <1-2 sentence answer>\n"
    "detailed answer: <comprehensive answer with key details>"
)

ASSISTANT_DELIMS = (
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "<|im_start|>assistant",
)
END_TOKENS = ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>")


@dataclass
class ProcessedResponse:
    query: str
    rag_informations: str | None
    rag_summary: str | None
    web_summary: str
    short_answer: str
    detailed_answer: str
    sources: Dict[str, Any]  # Maps URLs to lists of titles


def extract_response(content: str | list[str | dict]) -> str:
    response_content = content
    if isinstance(response_content, str):
        response = response_content
    else:
        response_tmp = response_content[-1]
        response_tmp: str | dict[str, str]

        if isinstance(response_tmp, str):
            response = response_tmp
        else:
            response = response_tmp.get("content", "")

    return response


class WebsearchPipeline:
    """
    Pipeline for running RAG and iterative websearch loops,
    integrating retrieved knowledge into enhanced answers.
    """

    def __init__(self, config: WebsearchConfig):
        self.config = config
        self.llm = self._initialize_llm()
        self._tokenizer = self._get_tokenizer()
        self._warned_fallback_tokenizer = False
        self.rag_results = None
        self.searcher = WebsearchOnly(
            provider=self.config.search_provider,
            max_results=self.config.max_searches,
            max_retries=self.config.max_retries,
        )

    def _initialize_llm(self) -> BaseChatModel:
        if self.config.use_rag:
            rag_cfg = self.config.access_rag_config()
            llm_conf: Dict[str, Any] = rag_cfg.get("rag", {}).get("llm")
            if llm_conf is None:
                raise ValueError(
                    "Missing 'llm' config under 'rag' in RAG configuration."
                )
            return LLM.from_config(LLMConfig(**llm_conf))
        else:
            base_conf = self.config.get_llm_config()
            base_conf = base_conf.__dict__
            return LLM.from_config(LLMConfig(**base_conf))

    def generate_summary(self, content: str | None, query: str):
        """Summarize content relevant to the query."""
        prefix = SUMMARY_PREFIX.format(query=query)
        fitted = self._fit_to_budget(
            content or "No context yet", SUMMARY_SYSTEM_MSG, prefix, SUMMARY_SUFFIX
        )
        prompt = prefix + fitted + SUMMARY_SUFFIX

        messages = [
            SystemMessage(content=SUMMARY_SYSTEM_MSG),
            HumanMessage(content=prompt),
        ]

        response_llm = self.llm.invoke(messages)
        response = extract_response(response_llm.content)

        return self._clean_llm_output(response)

    def evaluate_subquery_relevance(
        self, query, current_subqueries, previous_subqueries
    ):
        prompt = RELEVANCE_PROMPT.format(
            query=query,
            previous_subqueries=previous_subqueries,
            current_subqueries=current_subqueries,
        )
        messages = [
            SystemMessage(content=RELEVANCE_SYSTEM_MSG),
            HumanMessage(content=prompt),
        ]
        response_llm = self.llm.invoke(messages)
        response_content = extract_response(response_llm.content)
        response = self._clean_llm_output(response_content).strip().lower()
        if re.match(r"^yes\b", response):
            return True
        if re.match(r"^no\b", response):
            return False
        logger.warning(
            f"Unexpected LLM relevance response (expected 'yes'/'no'): '{response}'"
        )
        return False

    def _clean_llm_output(self, content: str) -> str:
        for delim in ASSISTANT_DELIMS:
            if delim in content:
                content = content.rsplit(delim, 1)[-1]
                break
        for end in END_TOKENS:
            content = content.replace(end, "")
        return content.strip()

    def _get_tokenizer(self):
        """Try to get a local tokenizer."""
        return getattr(self.llm, "tokenizer", None)

    def _encode(self, text: str) -> list[int]:
        """Encode text to token IDs using the llm tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("No tokenizer is available for encoding text.")
        return self._tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text using the llm tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("No tokenizer is available for decoding token IDs.")
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def _count_tokens(self, text: str) -> int:
        """Count tokens using heuristic, local tokenizer, or LLM."""
        if self.config.fast_tokenizer:
            return math.ceil(len(text) / 4)
        if self._tokenizer is not None:
            return len(self._encode(text))
        if not self._warned_fallback_tokenizer:
            logger.warning(
                "No local tokenizer available; token counts may be inaccurate. "
                "Consider setting fast_tokenizer=True in your config."
            )
            self._warned_fallback_tokenizer = True
        return self.llm.get_num_tokens(text)

    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within a token budget."""
        if self.config.fast_tokenizer:
            char_limit = max_tokens * 4
            if len(text) <= char_limit:
                return text
            return text[:char_limit]

        if self._tokenizer is not None:
            token_ids = self._encode(text)
            if len(token_ids) <= max_tokens:
                return text
            else:
                return self._decode(token_ids[:max_tokens])

        # Fallback when no local tokenizer: proportional char cut with a 10% safety
        # margin because uneven token/char ratios can produce results that are too long.
        total_tokens = self._count_tokens(text)
        if total_tokens <= max_tokens:
            return text
        ratio = max_tokens / total_tokens * 0.9
        cut = int(len(text) * ratio)
        return text[:cut] if cut > 0 else ""

    def _fit_to_budget(self, content: str, *fixed_parts: str) -> str:
        """Truncate content so that prompt fits within max_context_tokens."""
        fixed_tokens = sum(self._count_tokens(p) for p in fixed_parts)
        available = self.config.max_context_tokens - fixed_tokens
        if available <= 0:
            raise ValueError(
                "Prompt fixed parts exceed max_context_tokens: "
                f"max_context_tokens={self.config.max_context_tokens}, "
                f"fixed_tokens={fixed_tokens}. "
                "Reduce the fixed prompt size or increase max_context_tokens."
            )
        return self._truncate_to_token_limit(content, available)

    def generate_subqueries(
        self, original_query: str, current_context: Optional[str] = None
    ) -> List[str]:
        """
        Generate concise search subqueries
        """
        n = self.config.n_subqueries
        instruction = f"Question: {original_query}\n\n"
        if current_context is None:
            task = SUBQUERY_TASK.format(n=n)
        else:
            task = SUBQUERY_TASK_WITH_CONTEXT.format(
                n=n, current_context=current_context
            )

        prompt = instruction + task
        messages = [
            SystemMessage(content=SUBQUERY_SYSTEM_MSG),
            HumanMessage(content=prompt),
        ]

        response_llm = self.llm.invoke(messages)
        response = extract_response(response_llm.content)
        cleaned_answer = self._clean_llm_output(response)
        cleaned_answer = re.findall(
            r"subquery \d+: (.*)", cleaned_answer, flags=re.IGNORECASE
        )
        return cleaned_answer

    def web_search(self, query: str) -> List[Dict[str, str]]:
        """
        Perform a web search using the configured provider (WebsearchOnly).
        Includes exponential backoff retry logic to fix timeout issues (#230).
        Returns a list of dicts with keys: 'snippet', 'title' and 'url'
        """
        results = self.searcher.websearch_pipeline(query)
        return [
            {
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
                "title": r.get("title", ""),
            }
            for r in results
        ]

    def _compute_content_budget(self, *fixed_parts: str) -> int:
        """Compute how many tokens are available for content given fixed prompt parts."""
        fixed_tokens = sum(self._count_tokens(p) for p in fixed_parts)
        return max(0, self.config.max_context_tokens - fixed_tokens)

    def integrate_with_llm(
        self, original: str, rag_doc: str | None, web_content: str
    ) -> Dict[str, str]:
        rag_text = rag_doc or "No RAG sources"
        prefix = SYNTHESIS_PREFIX.format(original=original, rag_doc=rag_text)
        fitted = self._fit_to_budget(
            web_content, SYNTHESIS_SYSTEM_MSG, prefix, SYNTHESIS_SUFFIX
        )
        prompt = prefix + fitted + SYNTHESIS_SUFFIX

        msgs = [
            SystemMessage(content=SYNTHESIS_SYSTEM_MSG),
            HumanMessage(content=prompt),
        ]
        response_llm = self.llm.invoke(msgs)
        response = extract_response(response_llm.content)
        # parse
        clean_content = self._clean_llm_output(response)

        sa_matches = re.findall(
            r"short answer:\s*(.*?)(?=detailed answer:)",
            clean_content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        da_matches = re.findall(
            r"detailed answer:\s*(.*)", clean_content, flags=re.IGNORECASE | re.DOTALL
        )

        short = sa_matches[-1].strip().rstrip(",") if sa_matches else ""
        detailed = da_matches[-1].strip() if da_matches else ""
        return {"short": short, "detailed": detailed}

    def process_record(
        self,
        rec: Dict[str, Any],
        on_status: Optional[Callable[[str], None]] = None,
        on_excerpt: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        def status(msg: str) -> None:
            if on_status is not None:
                on_status(msg)

        def excerpt(title: str, snippet: str) -> None:
            if on_excerpt is not None:
                text = " ".join(f"{title}: {snippet}".split())
                on_excerpt(text[:90] + ("..." if len(text) > 90 else ""))

        qr = rec.get("input", "").strip()
        rag_ans = rec.get("answer", "") if self.config.use_rag else None
        self.rag_results = rag_ans
        rag_summary = (
            self.generate_summary(rag_ans, qr) if self.config.use_rag else None
        )

        source_map = {}
        seen_results = set()
        current_context = rag_summary
        final_short, final_detailed = "", ""
        web_summary = ""
        web_summary_all = ""  # will be reassigned later

        web_summaries = []
        previous_sub = []

        for loop in range(self.config.n_loops):
            tag = f"loop {loop + 1}/{self.config.n_loops}"
            status(f"{tag}: generating subqueries")
            if self.config.use_rag:
                subs = self.generate_subqueries(qr, current_context)
            else:
                subs = self.generate_subqueries(qr)  # Based on original query only

            snippets = []
            subquery_summaries = []

            if loop > 0 and not self.evaluate_subquery_relevance(
                qr, subs, previous_sub
            ):
                break

            # Build RAG context: RAG summary + prior loop answer when available
            rag_for_llm = rag_summary or ""
            if current_context and current_context != rag_summary:
                rag_for_llm += f"\n\nPrior answer:\n{current_context}"

            # Token-aware accumulation: use effective budget that accounts for
            # the fixed prompt overhead in the downstream prompt.
            # snippet_budget caps total snippets (for integrate_with_llm when not summarizing).
            if self.config.use_summary:
                snippet_budget = self.config.max_context_tokens
            else:
                synthesis_prefix = SYNTHESIS_PREFIX.format(
                    original=qr, rag_doc=rag_for_llm or "No RAG sources"
                )
                snippet_budget = self._compute_content_budget(
                    SYNTHESIS_SYSTEM_MSG, synthesis_prefix, SYNTHESIS_SUFFIX
                )
            total_tokens = 0
            budget_exhausted = False

            for sq in subs:
                if budget_exhausted:
                    break

                # Compute per-subquery summary budget using the current subquery
                sq_prefix = SUMMARY_PREFIX.format(query=sq)
                summary_budget = self._compute_content_budget(
                    SUMMARY_SYSTEM_MSG, sq_prefix, SUMMARY_SUFFIX
                )

                status(f"{tag}: searching web ({sq[:40]})")
                # Only sleep for DDG, and make it 2 seconds instead of 10
                if self.config.search_provider == "duckduckgo":
                    time.sleep(2)
                res = self.web_search(query=sq)

                subquery_snippets = []
                subquery_tokens = 0

                for r in res:
                    url = r["url"]
                    snippet = r["snippet"]
                    title = r["title"]

                    # Prevent duplicated results
                    if (url, snippet) in seen_results:
                        continue

                    snippet_tokens = self._count_tokens(snippet + "\n")

                    if total_tokens + snippet_tokens > snippet_budget:
                        logger.debug(
                            "Token budget reached (%d/%d tokens), skipping remaining searches on the web",
                            total_tokens,
                            snippet_budget,
                        )
                        budget_exhausted = True
                        break

                    if subquery_tokens + snippet_tokens > summary_budget:
                        break

                    if url not in source_map:
                        source_map[url] = []

                    if title not in source_map[url]:
                        source_map[url].append(title)

                    if not subquery_snippets:
                        excerpt(title, snippet)
                    snippets.append(snippet)
                    subquery_snippets.append(snippet)
                    total_tokens += snippet_tokens
                    subquery_tokens += snippet_tokens
                    seen_results.add((url, snippet))

                # Run this ONCE per subquery!
                if subquery_snippets:
                    combined_snippets = "\n".join(subquery_snippets)
                    summary = self.generate_summary(combined_snippets, sq)
                    subquery_summaries.append(summary)

            previous_sub = subs

            # Clear memory
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

            status(f"{tag}: summarizing results")
            combined_sub_summaries = "\n".join(
                [str(s) if s else "" for s in subquery_summaries]
            )
            web_summary = self.generate_summary(combined_sub_summaries, qr)
            web_summaries.append(web_summary)

            if self.config.use_summary:
                web_for_llm = web_summary
            else:
                web_for_llm = "\n".join(snippets)

            combined_web_summaries = "\n".join(
                [str(s) if s else "" for s in web_summaries]
            )
            web_summary_all = self.generate_summary(combined_web_summaries, qr)

            status(f"{tag}: synthesizing answer ({len(source_map)} sources)")
            out = self.integrate_with_llm(qr, rag_for_llm, web_for_llm)
            final_short, final_detailed = out["short"], out["detailed"]

            # Prepare context for next search loop
            current_context = final_detailed

        solution = ProcessedResponse(
            query=qr,
            rag_informations=self.rag_results,
            rag_summary=rag_summary if self.config.use_rag else None,
            web_summary=web_summary_all,
            short_answer=final_short,
            detailed_answer=final_detailed,
            sources=source_map,
        )

        return asdict(solution)

    def run(self):
        # RAG pipeline
        if self.config.use_rag:
            if not self.config.rag_config_path:
                raise ValueError("rag_config_path required when use_rag=True")
            logger.debug("Running RAG pipeline...")
            rag(self.config.rag_config_path)
            rc = self.config.access_rag_config()
            self.config.input_file = rc["mode_args"]["output_file"]

            assert self.config.input_file
            with open(self.config.input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            self.config.input_file = self.config.input_queries
            data = []

            assert self.config.input_file
            with open(self.config.input_file, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line.strip()))  # JSONL format

        config = self.config
        step_intro(
            "WebSearch",
            WEBSEARCH_EMOJI,
            "Answer each question by searching the web and your docs",
            [
                f"{len(data)} queries",
                f"provider: {config.search_provider}",
                f"<={config.n_loops} loops x {config.n_subqueries} subqueries",
                f"RAG: {'on' if config.use_rag else 'off'}",
            ],
        )

        start = time.time()
        time_loading = model_loading_seconds()
        outputs = []
        bar = progress(total=len(data), desc="Web search", unit="")
        for i, rec in enumerate(data, 1):
            question = str(rec.get("input", ""))
            bar.print_above(f"[{Color.BRAND}]Q{i}/{len(data)}[/] {escape(question)}")
            outputs.append(
                self.process_record(
                    rec,
                    on_status=bar.set_postfix_str,
                    on_excerpt=lambda e: bar.print_above(f"   [dim]{escape(e)}[/]"),
                )
            )
            bar.update(1)
        bar.close()

        # save
        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start - (model_loading_seconds() - time_loading)
        unique_sources = len({u for o in outputs for u in (o.get("sources") or {})})
        per_query = elapsed / len(outputs) if outputs else 0.0
        step_summary(
            "WebSearch",
            WEBSEARCH_EMOJI,
            elapsed,
            {
                "queries": len(outputs),
                "unique sources": unique_sources,
                "avg": f"{per_query:.0f} s/query",
                "output": str(output_path),
            },
        )

    def run_api(self, use_rag, use_summary, query):
        """
        Process queries and handle them with a temporary JSONL file.
        Parameters:
        - use_rag (bool): Indicates whether to use RAG.
        - use_summary (bool): Indicates whether to use summarization.
        - query (list): List of query dictionaries.
        Returns:
        - List of processed query results.
        """
        # Save query to a temporary JSONL file
        self.config.use_rag = use_rag
        self.config.use_summary = use_summary

        temp_file_path = self._save_query_as_json(query)

        try:
            outputs = []
            # Read from the temporary JSONL file
            with open(temp_file_path, "r", encoding="utf-8") as f:
                if self.config.use_rag:
                    for line in f:
                        record = json.loads(line)
                        outputs.append(self.process_record(record))
                else:
                    for line in f:
                        record = json.loads(line.strip())
                        outputs.append(self.process_record(record))

            return outputs

        finally:
            # Delete the temporary file
            logger.debug(f"Deleting temporary file: {temp_file_path}")
            os.remove(temp_file_path)

    def _save_query_as_json(self, query):
        """Save query to a temporary JSONL file and return the file path."""
        suffix = ".json" if self.config.use_rag else ".jsonl"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False
        ) as temp_file:
            # Convert Pydantic models to dictionaries if needed
            if isinstance(query, list):
                temp_file.writelines(
                    json.dumps(q.dict() if hasattr(q, "dict") else q) + "\n"
                    for q in query
                )
            else:
                temp_file.write(
                    json.dumps(query.dict() if hasattr(query, "dict") else query) + "\n"
                )
            logger.debug(f"Query saved to temporary file: {temp_file.name}")
            return temp_file.name
