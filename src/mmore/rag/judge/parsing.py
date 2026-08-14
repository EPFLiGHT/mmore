"""LLM response parsing and prompt formatting."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def parse_context_relevance_score(parsed: Dict[str, Any]) -> Optional[float]:
    raw = parsed.get("context_relevance_score")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid context_relevance_score %r", raw)
        return None


def repair_json_text(text: str) -> str:
    """Fix common invalid JSON from instruct models (trailing commas, Python literals)."""
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text


def extract_judge_fields_loose(text: str) -> Dict[str, Any]:
    """Best-effort field extraction when strict JSON parsing fails."""
    obj: Dict[str, Any] = {}
    if m := re.search(r'"decision"\s*:\s*"([A-Z_]+)"', text, re.IGNORECASE):
        obj["decision"] = m.group(1).upper()
    if m := re.search(r'"context_relevance_score"\s*:\s*([\d.]+)', text):
        obj["context_relevance_score"] = float(m.group(1))
    if m := re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL):
        obj["reason"] = m.group(1)
    if m := re.search(r'"sufficient"\s*:\s*(true|false)', text, re.IGNORECASE):
        obj["sufficient"] = m.group(1).lower() == "true"
    if m := re.search(r'"extra_questions"\s*:\s*(\[[^\]]*\])', text, re.DOTALL):
        try:
            extra = json.loads(m.group(1))
            if isinstance(extra, list):
                obj["extra_questions"] = [str(q) for q in extra]
        except json.JSONDecodeError:
            pass
    if m := re.search(r'"web_query"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        obj["web_query"] = m.group(1)
    elif re.search(r'"web_query"\s*:\s*null', text):
        obj["web_query"] = None
    if "decision" not in obj:
        raise json.JSONDecodeError("No decision field found", text, 0)
    return obj


def _judge_json_snippet(text: Any) -> str:
    """Isolate the judge JSON object (assistant reply, not prompt metrics)."""
    text = extract_llm_text(text).strip()
    if not text:
        return text

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "decision" in obj:
            return text[i : i + end]

    decision_matches = list(re.finditer(r'"decision"\s*:', text, re.IGNORECASE))
    if decision_matches:
        decision_pos = decision_matches[-1].start()
        start = text.rfind("{", 0, decision_pos)
        if start != -1:
            return text[start:]

    start = text.find("{")
    return text[start:] if start != -1 else text


def _parse_json_dict(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```.*$", "", text, flags=re.DOTALL).strip()

    start = text.find("{")
    if start == -1:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON object found", text, 0)
        start = match.start()

    snippet = text[start:]
    last_error: Optional[Exception] = None
    for candidate in (snippet, repair_json_text(snippet)):
        try:
            obj, _ = json.JSONDecoder().raw_decode(candidate)
            if not isinstance(obj, dict):
                raise TypeError(f"Expected JSON object, got {type(obj).__name__}")
            if "decision" not in obj:
                raise json.JSONDecodeError(
                    "Parsed JSON object has no decision field", candidate, 0
                )
            return obj
        except (json.JSONDecodeError, TypeError) as e:
            last_error = e
            continue

    try:
        return extract_judge_fields_loose(snippet)
    except json.JSONDecodeError:
        if last_error is not None:
            raise last_error
        raise json.JSONDecodeError("Could not parse judge JSON", snippet, 0)


def parse_judge_llm_response(content: Any) -> tuple[str, Dict[str, Any]]:
    """Extract the judge JSON snippet and parse it (single preprocessing pass)."""
    raw = _judge_json_snippet(content)
    return raw, _parse_json_dict(raw)


def extract_judge_raw_response(content: Any) -> str:
    """Judge JSON snippet for logging when full parsing fails."""
    return _judge_json_snippet(content)


def parse_json_response(text: str) -> Dict[str, Any]:
    return _parse_json_dict(_judge_json_snippet(text))


_LLAMA_ASSISTANT_DELIMITERS = (
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "<|eot_id|>assistant",
    "<|assistant|>",
)
_LLAMA_END_MARKERS = ("<|eot_id|>", "<|end_of_text|>")


def extract_llm_text(content: Any) -> str:
    if isinstance(content, list):
        content = content[-1] if content else ""
    if isinstance(content, dict):
        content = content.get("content", "")
    text = str(content).strip()
    for delimiter in _LLAMA_ASSISTANT_DELIMITERS:
        if delimiter in text:
            text = text.split(delimiter, 1)[-1]
            break
    for end in _LLAMA_END_MARKERS:
        if end in text:
            text = text.split(end, 1)[0]
    return text.strip()


def format_chunks_for_prompt(docs: List[Document], max_chunk_chars: int) -> str:
    lines = []
    for doc in docs:
        content = doc.page_content
        if len(content) > max_chunk_chars:
            content = content[:max_chunk_chars] + "..."
        sim = doc.metadata.get("similarity", "n/a")
        rerank = doc.metadata.get("rerank_score", "n/a")
        rank = doc.metadata.get("rank", "?")
        lines.append(
            f"[rank={rank}, similarity={sim}, rerank_score={rerank}] {content}"
        )
    return "\n".join(lines) if lines else "(no chunks retrieved)"


def effective_retrieve_params(
    retrieve_params: Optional[Dict[str, Any]],
    original_query: str,
    retriever_k: int,
) -> Dict[str, Any]:
    """Resolve RE_RETRIEVE params actually passed to the retriever."""
    params = retrieve_params or {}
    effective: Dict[str, Any] = {"input": params.get("input") or original_query}
    if params.get("k") is not None:
        effective["k"] = int(params["k"])
    else:
        effective["k"] = max(retriever_k * 2, retriever_k + 3)
    return effective
