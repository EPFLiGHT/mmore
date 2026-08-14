"""Decision coercion and step trace serialization."""

import logging
from typing import Any, Dict, List, Tuple

from .parsing import effective_retrieve_params
from .types import JudgeConfig, JudgeDecision, JudgeResult

logger = logging.getLogger(__name__)


def allowed_actions(config: JudgeConfig) -> List[str]:
    allowed = [JudgeDecision.PROCEED.value]
    if config.allow_add_questions:
        allowed.append(JudgeDecision.ADD_QUESTIONS.value)
    if config.allow_add_context:
        allowed.append(JudgeDecision.ADD_CONTEXT.value)
    if config.allow_re_retrieve:
        allowed.append(JudgeDecision.RE_RETRIEVE.value)
    return allowed


def coerce_decision(raw: str, config: JudgeConfig) -> Tuple[JudgeDecision, bool, str]:
    """Returns (final_decision, coerced, raw_decision)."""
    raw_decision = raw
    try:
        decision = JudgeDecision(raw)
    except ValueError:
        logger.warning("Unknown judge decision '%s', defaulting to PROCEED", raw)
        return JudgeDecision.PROCEED, True, raw_decision

    allowed = allowed_actions(config)
    if decision.value in allowed:
        return decision, False, raw_decision

    if decision != JudgeDecision.PROCEED:
        if decision == JudgeDecision.RE_RETRIEVE and config.allow_add_questions:
            logger.warning(
                "Judge chose disallowed action %s, falling back to ADD_QUESTIONS",
                decision.value,
            )
            return JudgeDecision.ADD_QUESTIONS, True, raw_decision
        if decision != JudgeDecision.RE_RETRIEVE and config.allow_re_retrieve:
            logger.warning(
                "Judge chose disallowed action %s, falling back to RE_RETRIEVE",
                decision.value,
            )
            return JudgeDecision.RE_RETRIEVE, True, raw_decision

    logger.warning(
        "Judge chose disallowed action %s, defaulting to PROCEED", decision.value
    )
    return JudgeDecision.PROCEED, True, raw_decision


def step_record(
    step: int,
    result: JudgeResult,
    query: str,
    retriever_k: int,
) -> Dict[str, Any]:
    """Serializable judge trace for judge_steps and retrieval_corrections."""
    record: Dict[str, Any] = {
        "step": step,
        "decision": result.decision.value,
        "raw_decision": result.raw_decision,
        "coerced_decision": result.coerced_decision,
        "exit_reason": result.exit_reason,
        "llm_invoked": result.llm_invoked,
        "context_relevance_score": result.context_relevance_score,
    }
    if result.llm_invoked and result.raw_llm_response:
        record["llm_response"] = result.raw_llm_response
    if result.decision == JudgeDecision.ADD_QUESTIONS:
        record["extra_questions"] = list(result.extra_questions)
    elif result.decision == JudgeDecision.ADD_CONTEXT:
        record["web_query"] = result.web_query
        record["effective_web_query"] = result.web_query or query
    elif result.decision == JudgeDecision.RE_RETRIEVE:
        record["retrieve_params"] = result.retrieve_params
        record["effective_retrieve_params"] = effective_retrieve_params(
            result.retrieve_params, query, retriever_k
        )
    return record
