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
from typing import List, Optional, Union

import dspy
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import Self

from ...rag.llm import LLMConfig
from ...utils import load_config
from ..config import PrivacyConfig, VerifierCheck
from ..dspy_llm import build_dspy_lm
from ..verification import CLEAN_VERDICT, VerifierVerdict, VerifierWarning, WarningKind
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


def _build_residual_predictor() -> dspy.Predict:
    return dspy.Predict(
        _ResidualLeakageSignature.with_instructions(_RESIDUAL_INSTRUCTION)
    )


def _build_faithfulness_predictor() -> dspy.Predict:
    return dspy.Predict(
        _FaithfulnessSignature.with_instructions(_FAITHFULNESS_INSTRUCTION)
    )


def _clamp_confidence(value: object) -> float:
    """Coerce a model-provided confidence into ``[0.0, 1.0]``, 0.0 on failure."""
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    try:
        return max(0.0, min(1.0, float(str(value))))
    except (TypeError, ValueError):
        return 0.0


def _clean_flagged(value: object) -> str | None:
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

    def __init__(
        self,
        config: PrivacyConfig,
        llm_config: Optional[LLMConfig] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self._dspy_lm: Optional[dspy.BaseLM] = None
        self._verifier_cfg = config.verifier
        super().__init__(config, llm_config=llm_config, checkpointer=checkpointer)

    @classmethod
    def from_config(
        cls,
        config: Union[PrivacyConfig, str, dict],
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> Self:
        if not isinstance(config, PrivacyConfig):
            config = load_config(config, PrivacyConfig)
        return cls(config, config.verifier.llm, checkpointer=checkpointer)

    @property
    def checks(self) -> List[VerifierCheck]:
        return list(self._verifier_cfg.checks)

    @property
    def warn_threshold(self) -> float:
        return self._verifier_cfg.warn_threshold

    def _ensure_dspy_lm(self) -> dspy.BaseLM:
        if self._dspy_lm is None:
            if self._llm_config is None:
                raise ValueError(
                    "Advisory verifier requires an LLM to check the answer."
                )
            self._dspy_lm = build_dspy_lm(self._llm_config)
        return self._dspy_lm

    def _predict(self, predictor: dspy.Predict, **inputs) -> dspy.Prediction | None:
        try:
            with dspy.context(lm=self._ensure_dspy_lm()):
                return predictor(**inputs)
        except Exception as e:
            logger.warning("Advisory check failed (%s), treating as no warning", e)
            return None

    def _check_residual_leakage(
        self, answer: str, sanitized_context: str, raw_context: str
    ) -> VerifierWarning | None:
        prediction = self._predict(
            _build_residual_predictor(),
            answer=answer,
            sanitized_context=sanitized_context,
            raw_context=raw_context,
        )
        if prediction is None:
            return None
        confidence = _clamp_confidence(getattr(prediction, "confidence", 0.0))
        if not getattr(prediction, "leaked", False) or confidence < self.warn_threshold:
            return None
        return VerifierWarning(
            kind=WarningKind.RESIDUAL_LEAKAGE,
            flagged=_clean_flagged(getattr(prediction, "entity_type", "")),
            evidence=str(getattr(prediction, "evidence", "")).strip(),
            confidence=confidence,
        )

    def _check_faithfulness(self, answer: str, evidence: str) -> VerifierWarning | None:
        prediction = self._predict(
            _build_faithfulness_predictor(), answer=answer, evidence=evidence
        )
        if prediction is None:
            return None
        confidence = _clamp_confidence(getattr(prediction, "confidence", 0.0))
        if (
            not getattr(prediction, "unfaithful", False)
            or confidence < self.warn_threshold
        ):
            return None
        return VerifierWarning(
            kind=WarningKind.FAITHFULNESS,
            flagged=_clean_flagged(getattr(prediction, "unsupported_claim", "")),
            evidence=str(getattr(prediction, "rationale", "")).strip(),
            confidence=confidence,
        )

    def verify(
        self, answer: str, sanitized_chunks: List[str], raw_chunks: List[str]
    ) -> VerifierVerdict:
        """Run the configured checks over the answer and the whole context."""
        if not answer.strip() or not self.checks:
            return CLEAN_VERDICT

        sanitized_context = "\n\n".join(c for c in sanitized_chunks if c).strip()
        raw_context = "\n\n".join(c for c in raw_chunks if c).strip()

        warnings: List[VerifierWarning] = []
        if VerifierCheck.RESIDUAL_LEAKAGE in self.checks:
            w = self._check_residual_leakage(answer, sanitized_context, raw_context)
            if w is not None:
                warnings.append(w)
        if VerifierCheck.FAITHFULNESS in self.checks:
            w = self._check_faithfulness(answer, raw_context)
            if w is not None:
                warnings.append(w)
        return VerifierVerdict(warnings=warnings)

    def _node(self, state: PrivacyState) -> PrivacyState:
        """Graph node: annotate the answer with the advisory verdict only."""
        verdict = self.verify(
            state.get("answer", ""),
            list(state.get("sanitized_chunks", [])),
            list(state.get("raw_chunks", [])),
        )
        return PrivacyState(verifier_verdict=verdict)
