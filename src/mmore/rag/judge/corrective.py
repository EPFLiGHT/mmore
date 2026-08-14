"""Corrective retrieval loop: apply actions and retrieve_with_judge."""

import logging
from typing import Any, Callable, Dict, List, Union, cast

from langchain_core.documents import Document

from ..retriever import Retriever
from .decisions import step_record
from .evaluator import LLMJudge
from .metrics import (
    log_correction_metrics,
    merge_documents,
    metrics_for_output,
    record_correction_metrics,
)
from .parsing import effective_retrieve_params
from .types import JudgeConfig, JudgeDecision, JudgeResult

logger = logging.getLogger(__name__)

RetrieverInput = Union[str, Dict[str, Any]]


def _invoke_retriever(
    retriever: Retriever,
    query_or_state: RetrieverInput,
    **kwargs: Any,
) -> List[Document]:
    """Invoke retriever with str or RAG state dict (LangChain stubs expect str)."""
    invoke = cast(Callable[..., List[Document]], retriever.invoke)
    return invoke(query_or_state, **kwargs)


def apply_corrective_action(
    retriever: Retriever,
    config: JudgeConfig,
    state: Dict[str, Any],
    docs: List[Document],
    result: JudgeResult,
) -> List[Document]:
    query = state["input"]

    if result.decision == JudgeDecision.ADD_QUESTIONS:
        for sub_query in result.extra_questions[:3]:
            sub_state = {**state, "input": sub_query}
            sub_docs = _invoke_retriever(retriever, sub_state)
            docs = merge_documents(docs, sub_docs)

    elif result.decision == JudgeDecision.ADD_CONTEXT:
        web_query = result.web_query or query
        web_docs = retriever._get_web_documents(
            web_query, max_results=config.max_web_results
        )
        docs = merge_documents(docs, web_docs)

    elif result.decision == JudgeDecision.RE_RETRIEVE:
        effective = effective_retrieve_params(
            result.retrieve_params, query, retriever.k
        )
        new_state = {**state, "input": effective["input"]}
        new_docs = _invoke_retriever(retriever, new_state, k=effective["k"])
        docs = merge_documents(docs, new_docs)

    if config.rerank_after_merge and docs:
        if getattr(retriever, "reranker_model", None):
            docs = retriever.rerank(query, docs)
        else:
            logger.info(
                "Judge rerank_after_merge requested but retriever has no reranker_model"
            )

    return docs


def retrieve_with_judge(
    retriever: Retriever,
    judge: LLMJudge,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run retrieval with optional corrective loop (retrieve -> judge -> action).

    Returns state enriched with docs, retrieval_metrics, judge_decision, judge_actions.
    """
    docs = _invoke_retriever(retriever, state)
    judge_actions: List[str] = []
    retrieval_corrections: List[Dict[str, Any]] = []
    judge_llm_calls = 0
    judge_steps: List[Dict[str, Any]] = []
    hit_max_steps = False
    config = judge.config
    query = state["input"]
    thresholds = config.metric_thresholds
    final_result = JudgeResult(decision=JudgeDecision.PROCEED, reason="")

    for step in range(config.max_corrective_steps + 1):
        final_result = judge.evaluate(
            query,
            docs,
            after_correction=bool(judge_actions),
            corrective_actions_used=len(judge_actions),
        )
        if final_result.llm_invoked:
            judge_llm_calls += 1

        judge_steps.append(step_record(step, final_result, query, retriever.k))
        logger.info(
            "Judge step %s | query=%r | decision=%s | context_relevance_score=%s | reason=%s",
            step,
            query[:120],
            final_result.decision.value,
            final_result.context_relevance_score,
            final_result.reason,
        )

        if final_result.decision == JudgeDecision.PROCEED:
            break

        if step >= config.max_corrective_steps:
            hit_max_steps = True
            logger.info(
                "Max corrective steps (%s) reached; proceeding with current docs",
                config.max_corrective_steps,
            )
            final_result = JudgeResult.proceed(
                "max_corrective_steps",
                context_relevance_score=final_result.context_relevance_score,
            )
            break

        action = final_result.decision.value
        judge_actions.append(action)
        docs_before = docs
        docs = apply_corrective_action(
            retriever, config, state, docs_before, final_result
        )
        correction = record_correction_metrics(
            action,
            docs_before,
            docs,
            thresholds,
            final_result.context_relevance_score,
        )
        retrieval_corrections.append(
            {
                **correction.to_dict(),
                **step_record(step, final_result, query, retriever.k),
            }
        )
        log_correction_metrics(query, correction)

    retrieval_metrics = metrics_for_output(
        docs, thresholds, final_result.context_relevance_score
    )
    logger.info(
        "Judge done | query=%r | decision=%s | actions=%s | thresholds_met=%s | "
        "context_relevance_score=%s | reason=%s",
        query[:120],
        final_result.decision.value,
        judge_actions,
        retrieval_metrics.thresholds_met,
        retrieval_metrics.context_relevance_score,
        final_result.reason,
    )

    out: Dict[str, Any] = {
        **state,
        "docs": docs,
        "retrieval_metrics": retrieval_metrics.to_dict(),
        "judge_decision": final_result.decision.value,
        "judge_reason": final_result.exit_reason,
        "judge_actions": judge_actions,
        "judge_llm_calls": judge_llm_calls,
        "judge_steps": judge_steps,
        "hit_max_corrective_steps": float(hit_max_steps),
    }
    if retrieval_corrections:
        out["retrieval_corrections"] = retrieval_corrections
    return out
