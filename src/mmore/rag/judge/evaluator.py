"""LLM-as-a-Judge evaluator."""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .decisions import allowed_actions, coerce_decision
from .metrics import evaluate_metrics
from .parsing import (
    extract_judge_raw_response,
    format_chunks_for_prompt,
    parse_context_relevance_score,
    parse_judge_llm_response,
)
from .types import JudgeConfig, JudgeDecision, JudgeResult

logger = logging.getLogger(__name__)


class LLMJudge:
    """LLM-as-a-Judge: structured JSON decision on retrieval quality."""

    def __init__(self, llm: BaseChatModel, config: JudgeConfig):
        self.llm = llm
        self.config = config

    def _early_exit(
        self,
        docs: List[Document],
        *,
        after_correction: bool,
    ) -> Optional[JudgeResult]:
        """Threshold pass or forced corrective action; None means invoke the LLM."""
        thresholds = self.config.metric_thresholds
        metrics, passed, status = evaluate_metrics(docs, thresholds)

        if passed:
            exit_reason = (
                "metrics_after_correction"
                if after_correction
                else "metrics_above_thresholds"
            )
            return JudgeResult.proceed(exit_reason)

        if self.config.force_corrective_action:
            forced = JudgeDecision(self.config.force_corrective_action)
            allowed = allowed_actions(self.config)
            if forced.value not in allowed:
                raise ValueError(
                    f"force_corrective_action {forced.value!r} not in allowed {allowed}"
                )
            retrieve_params = {"k": 10} if forced == JudgeDecision.RE_RETRIEVE else None
            return JudgeResult(
                decision=forced,
                reason="force_corrective_action",
                exit_reason="force_corrective_action",
                llm_invoked=False,
                raw_decision=forced.value,
                coerced_decision=False,
                retrieve_params=retrieve_params,
            )

        return None

    def _result_from_parsed(
        self,
        parsed: Dict[str, Any],
        *,
        llm_invoked: bool = True,
        raw_llm_response: Optional[str] = None,
    ) -> JudgeResult:
        decision, coerced, raw = coerce_decision(
            str(parsed.get("decision", JudgeDecision.PROCEED.value)), self.config
        )
        extra_questions = parsed.get("extra_questions") or []
        if not isinstance(extra_questions, list):
            extra_questions = []

        context_relevance_score = parse_context_relevance_score(parsed)
        exit_reason = (
            "llm_proceed" if decision == JudgeDecision.PROCEED else "llm_corrective"
        )

        return JudgeResult(
            decision=decision,
            reason=str(parsed.get("reason", "")),
            exit_reason=exit_reason,
            llm_invoked=llm_invoked,
            raw_decision=raw,
            coerced_decision=coerced,
            context_relevance_score=context_relevance_score,
            extra_questions=[str(q) for q in extra_questions],
            web_query=parsed.get("web_query"),
            retrieve_params=parsed.get("retrieve_params"),
            raw_llm_response=raw_llm_response,
        )

    def evaluate(
        self,
        query: str,
        docs: List[Document],
        *,
        after_correction: bool = False,
        corrective_actions_used: int = 0,
    ) -> JudgeResult:
        early = self._early_exit(docs, after_correction=after_correction)
        if early is not None:
            return early

        thresholds = self.config.metric_thresholds
        metrics, _, metrics_status = evaluate_metrics(docs, thresholds)
        max_steps = self.config.max_corrective_steps
        user_prompt = self.config.user_prompt.format(
            query=query,
            metrics=metrics.to_dict(),
            metrics_status=metrics_status,
            thresholds=thresholds,
            allowed_actions=allowed_actions(self.config),
            chunks=format_chunks_for_prompt(docs, self.config.max_chunk_chars),
            correction_step=corrective_actions_used + 1,
            corrective_actions_used=corrective_actions_used,
            max_corrective_steps=max_steps,
            remaining_corrective_steps=max(0, max_steps - corrective_actions_used),
            after_correction=after_correction,
        )
        response = self.llm.invoke(
            [
                SystemMessage(content=self.config.system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        try:
            raw_llm_response, parsed = parse_judge_llm_response(response.content)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                "Failed to parse judge JSON: %s | see judge_steps[].llm_response in output",
                e,
            )
            return JudgeResult.proceed(
                "parse_error_fallback",
                llm_invoked=True,
                raw_llm_response=extract_judge_raw_response(response.content),
            )

        return self._result_from_parsed(parsed, raw_llm_response=raw_llm_response)
