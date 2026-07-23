"""Post-cloud Advisory Verifier.

Pipeline:  ... -> gate -> answer -> [advisory_verifier] -> report
Reads:     answer, sanitized_chunks, raw_chunks
Writes:    verifier_verdict

The second specialized adversary, sibling to the pre-cloud leakage adversary.
Unlike the answer model (which only sees the sanitized context), the verifier
sees the whole context: the raw retrieved chunks, the sanitized context, and
the model's answer. It runs the configured advisory checks over the answer:

  - residual leakage: does the answer reintroduce PII or quasi-identifiers that
  sanitization had removed?
  - faithfulness: does the answer make claims unsupported by the evidence?
"""

import logging
from typing import Callable

import dspy
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import Self

from ...rag.llm import LLMConfig
from ..config import PrivacyConfig, VerifierCheck, as_privacy_config
from ..detection.constants import DEFAULT_LLM_CONFIG
from ..dspy_llm import clamp_confidence
from ..schemas.verification import CLEAN_VERDICT, VerifierVerdict, VerifierWarning
from ..ux import notify
from .base import BaseAgent
from .state import PrivacyState

logger = logging.getLogger(__name__)


# ========================================================================
# DSPy signatures
# ========================================================================

_ANSWER_DESC = "the model's answer shown to the user"
_SANITIZED_DESC = "the sanitized context the answer model was allowed to see"
_RAW_DESC = "the raw retrieved context, including the original personal values"
_EVIDENCE_CTX_DESC = "the retrieved evidence the answer must be grounded in"
_CONFIDENCE_DESC = "0.0-1.0 likelihood that the finding is real"

_LEAKED_DESC = "true if the answer reintroduces a protected or quasi-identifier"
_LEAK_ENTITY_DESC = "the leaked entity type, or NONE if the answer is clean"
_LEAK_EVIDENCE_DESC = (
    "brief justification citing the residual signal without restating any raw "
    "personal value"
)

_UNFAITHFUL_DESC = "true if the answer makes a claim the evidence does not support"
_CLAIM_DESC = "the unsupported or hallucinated claim, or NONE if the answer is faithful"
_FAITH_RATIONALE_DESC = (
    "brief justification: what the evidence does or does not support"
)

_RESIDUAL_INSTRUCTION = (
    "You are a privacy verifier. The answer was produced from the sanitized "
    "context only. Decide whether the answer nonetheless reintroduces personal "
    "or quasi-identifying information, by regurgitation or by inference, that "
    "the sanitized context had removed. Use the raw context to judge what was "
    "protected. Be strict. Do not echo raw personal values in your evidence."
)

_FAITHFULNESS_INSTRUCTION = (
    "You are a faithfulness verifier. Cross-check the answer against the "
    "retrieved evidence and decide whether every claim it makes is supported. "
    "Flag hallucinations and unsupported claims. Be strict: a claim with no "
    "grounding in the evidence is unfaithful."
)


class _ResidualLeakageSignature(dspy.Signature):
    answer: str = dspy.InputField(desc=_ANSWER_DESC)
    sanitized_context: str = dspy.InputField(desc=_SANITIZED_DESC)
    raw_context: str = dspy.InputField(desc=_RAW_DESC)
    leaked: bool = dspy.OutputField(desc=_LEAKED_DESC)
    entity_type: str = dspy.OutputField(desc=_LEAK_ENTITY_DESC)
    evidence: str = dspy.OutputField(desc=_LEAK_EVIDENCE_DESC)
    confidence: float = dspy.OutputField(desc=_CONFIDENCE_DESC)


class _FaithfulnessSignature(dspy.Signature):
    answer: str = dspy.InputField(desc=_ANSWER_DESC)
    evidence: str = dspy.InputField(desc=_EVIDENCE_CTX_DESC)
    unfaithful: bool = dspy.OutputField(desc=_UNFAITHFUL_DESC)
    unsupported_claim: str = dspy.OutputField(desc=_CLAIM_DESC)
    rationale: str = dspy.OutputField(desc=_FAITH_RATIONALE_DESC)
    confidence: float = dspy.OutputField(desc=_CONFIDENCE_DESC)


def _flagged_or_none(value: object) -> str | None:
    """Normalize a flagged entity/claim, treating empty or NONE as nothing."""
    text = str(value).strip()
    return text if text and text.upper() != "NONE" else None


# ========================================================================
# Agent
# ========================================================================


class AdvisoryVerifierAgent(BaseAgent):
    """Post-cloud advisory verifier over the answer and the whole context."""

    state_schema = PrivacyState
    node_name = "advisory_verifier"
    fallback_llm_config = DEFAULT_LLM_CONFIG

    def __init__(
        self,
        config: PrivacyConfig,
        llm_config: LLMConfig | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self._verifier_cfg = config.verifier
        super().__init__(config, llm_config=llm_config, checkpointer=checkpointer)

    @classmethod
    def from_config(
        cls,
        config: PrivacyConfig | str | dict,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> Self:
        config = as_privacy_config(config)
        llm_config = config.verifier.llm
        if llm_config is None and config.context_analyzer:
            llm_config = config.context_analyzer.llm
        return cls(config, llm_config, checkpointer=checkpointer)

    @property
    def checks(self) -> list[VerifierCheck]:
        return list(self._verifier_cfg.checks)

    @property
    def warn_threshold(self) -> float:
        return self._verifier_cfg.warn_threshold

    def _predict(
        self, signature: type[dspy.Signature], instruction: str, **inputs
    ) -> dspy.Prediction:
        with dspy.context(lm=self.dspy_lm):
            return dspy.Predict(signature.with_instructions(instruction))(**inputs)

    def _check_residual_leakage(
        self, answer: str, sanitized_context: str, raw_context: str
    ) -> VerifierWarning | None:
        prediction = self._predict(
            _ResidualLeakageSignature,
            _RESIDUAL_INSTRUCTION,
            answer=answer,
            sanitized_context=sanitized_context,
            raw_context=raw_context,
        )
        confidence = clamp_confidence(getattr(prediction, "confidence", 0.0))
        if not getattr(prediction, "leaked", False) or confidence < self.warn_threshold:
            return None
        return VerifierWarning(
            kind=VerifierCheck.RESIDUAL_LEAKAGE,
            flagged=_flagged_or_none(getattr(prediction, "entity_type", "")),
            evidence=str(getattr(prediction, "evidence", "")).strip(),
            confidence=confidence,
        )

    def _check_faithfulness(self, answer: str, evidence: str) -> VerifierWarning | None:
        prediction = self._predict(
            _FaithfulnessSignature,
            _FAITHFULNESS_INSTRUCTION,
            answer=answer,
            evidence=evidence,
        )
        confidence = clamp_confidence(getattr(prediction, "confidence", 0.0))
        if (
            not getattr(prediction, "unfaithful", False)
            or confidence < self.warn_threshold
        ):
            return None
        return VerifierWarning(
            kind=VerifierCheck.FAITHFULNESS,
            flagged=_flagged_or_none(getattr(prediction, "unsupported_claim", "")),
            evidence=str(getattr(prediction, "rationale", "")).strip(),
            confidence=confidence,
        )

    def verify(
        self, answer: str, sanitized_chunks: list[str], raw_chunks: list[str]
    ) -> VerifierVerdict:
        """Run the configured checks over the answer and the whole context."""
        if not answer.strip() or not self.checks:
            return CLEAN_VERDICT

        sanitized_context = "\n\n".join(c for c in sanitized_chunks if c).strip()
        raw_context = "\n\n".join(c for c in raw_chunks if c).strip()
        runners: dict[VerifierCheck, Callable[[], VerifierWarning | None]] = {
            VerifierCheck.RESIDUAL_LEAKAGE: lambda: self._check_residual_leakage(
                answer, sanitized_context, raw_context
            ),
            VerifierCheck.FAITHFULNESS: lambda: self._check_faithfulness(
                answer, raw_context
            ),
        }

        warnings: list[VerifierWarning] = []
        ran: list[str] = []
        failed: list[str] = []
        for check in self.checks:
            run = runners.get(check)
            if run is None:
                continue
            try:
                warning = run()
            except Exception as e:
                logger.debug("Verifier check %s failed: %s", check.value, e)
                notify(
                    f"Verifier: the {check.value} check could not run, "
                    "its result is unknown for this answer",
                    logger,
                )
                failed.append(check.value)
                continue
            ran.append(check.value)
            if warning is not None:
                warnings.append(warning)
        return VerifierVerdict(warnings=warnings, checks_run=ran, checks_failed=failed)

    def _node(self, state: PrivacyState) -> PrivacyState:
        """Graph node: annotate the answer with the advisory verdict only."""
        verdict = self.verify(
            state.get("answer", ""),
            list(state.get("sanitized_chunks", [])),
            list(state.get("raw_chunks", [])),
        )
        return PrivacyState(verifier_verdict=verdict)
