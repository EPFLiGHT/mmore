"""Pre-cloud Leakage Adversary.

Pipeline:  analyzer -> detector -> sanitizer -> [leakage_adversary] -> gate
Reads:     policy, sanitized_chunks
Writes:    verdict, safe

The trust-boundary probe: it attacks the sanitized context for residual PII and
quasi-identifiers before anything leaves for the cloud answer model. It runs one
adversarial probe per configured attack vector, keeps the strongest signal, and
treats a probe whose confidence reaches the threashold as a leak.
"""

import logging
from typing import List, Optional, Union

import dspy
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import Self

from ...rag.llm import LLMConfig
from ...utils import load_config
from ..config import AttackVector, PrivacyConfig
from ..dspy_llm import build_dspy_lm
from ..leakage import LeakageVerdict
from ..policy import PrivacyPolicy
from .base import BaseAgent
from .state import PrivacyState

logger = logging.getLogger(__name__)


# ========================================================================
# Attack taxonomy (Guan et al. 2025 Section 3.2.3)
# ========================================================================

_VECTOR_GUIDANCE = {
    AttackVector.RESIDUAL_SPAN: (
        "Residual-span extraction: scan for any protected identifier the "
        "sanitizer left verbatim or only partially masked (a name, number, "
        "email, date, or location still readable in the clear)."
    ),
    AttackVector.QUASI_IDENTIFIER: (
        "Quasi-identifier synthesis: combine individually non-identifying "
        "attributes (a rare attribute + a location + a date) that together "
        "single out one person even though no direct identifier remains."
    ),
    AttackVector.STRUCTURAL_REID: (
        "Structural re-identification: exploit the uniqueness of a record's "
        "structure (an unusual combination of fields or an outlier value) to "
        "pick out a specific individual from the population."
    ),
    AttackVector.CONTEXT_RECONSTRUCTION: (
        "Context reconstruction: infer what a redacted or masked span must "
        "have been from the retained surrounding text (e.g. a placeholder "
        "that the remaining context uniquely determines)."
    ),
    AttackVector.MEMBERSHIP_INFERENCE: (
        "Membership inference: decide whether a specific, nameable individual "
        "is present in this context from the retained details, even without "
        "recovering a direct identifier."
    ),
}


# ========================================================================
# DSPy signature
# ========================================================================

_CONTEXT_DESC = "the sanitized retrieved context to attack"
_ENTITIES_DESC = "entity types that must not be recoverable from the context"
_ATTACK_DESC = "the attack vector to attempt and how it works"
_LEAKED_DESC = "true if the attack recovers or infers a protected identifier"
_ENTITY_TYPE_DESC = "the entity type that leaks, or NONE if the attack fails"
_EVIDENCE_DESC = (
    "brief justification for the verdict, citing the residual signal without "
    "restating any raw personal value"
)
_CONFIDENCE_DESC = "0.0-1.0 likelihood that the attack succeeds"

_PROBE_INSTRUCTION = (
    "You are a privacy red-team adversary. Attempt the described attack "
    "against the sanitized context and report whether a protected identifier "
    "can be recovered or inferred. Be strict: a residual or reconstructable "
    "identifier is a leak. Do not echo raw personal values in your evidence."
)


class _LeakageProbeSignature(dspy.Signature):
    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    sensitive_entities: List[str] = dspy.InputField(desc=_ENTITIES_DESC)
    attack: str = dspy.InputField(desc=_ATTACK_DESC)
    leaked: bool = dspy.OutputField(desc=_LEAKED_DESC)
    entity_type: str = dspy.OutputField(desc=_ENTITY_TYPE_DESC)
    evidence: str = dspy.OutputField(desc=_EVIDENCE_DESC)
    confidence: float = dspy.OutputField(desc=_CONFIDENCE_DESC)


def _build_probe_predictor() -> dspy.Predict:
    return dspy.Predict(_LeakageProbeSignature.with_instructions(_PROBE_INSTRUCTION))


def _clamp_confidence(value: object) -> float:
    """Coerce a model-provided confidence into ``[0.0, 1.0]``, 0.0 on failure."""
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    try:
        return max(0.0, min(1.0, float(str(value))))
    except (TypeError, ValueError):
        return 0.0


_SAFE_VERDICT = LeakageVerdict(
    leaked=False, vector="none", entity_type="NONE", evidence="", confidence=0.0
)


# ========================================================================
# Agent
# ========================================================================


class AdversarialAgent(BaseAgent):
    """Adversarially probes the sanitized context for residual leakage."""

    state_schema = PrivacyState
    node_name = "leakage_adversary"

    def __init__(
        self,
        config: PrivacyConfig,
        llm_config: Optional[LLMConfig] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self._dspy_lm: Optional[dspy.BaseLM] = None
        self._adversary_cfg = config.leakage_adversary
        super().__init__(config, llm_config=llm_config, checkpointer=checkpointer)

    @classmethod
    def from_config(
        cls,
        config: Union[PrivacyConfig, str, dict],
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> Self:
        if not isinstance(config, PrivacyConfig):
            config = load_config(config, PrivacyConfig)
        llm_config = config.leakage_adversary.llm
        return cls(config, llm_config, checkpointer=checkpointer)

    @property
    def strategies(self) -> List[AttackVector]:
        return list(self._adversary_cfg.strategies)

    @property
    def leakage_threshold(self) -> float:
        return self._adversary_cfg.leakage_threshold

    def _ensure_dspy_lm(self) -> dspy.BaseLM:
        if self._dspy_lm is None:
            if self._llm_config is None:
                raise ValueError("Leakage adversary requires an LLM to probe leakage.")
            self._dspy_lm = build_dspy_lm(self._llm_config)
        return self._dspy_lm

    def _probe_vector(
        self,
        predictor: dspy.Predict,
        context: str,
        entities: List[str],
        vector: AttackVector,
    ) -> LeakageVerdict:
        """Run one attack vector; confidence 0.0 if the probe errors out."""
        try:
            with dspy.context(lm=self._ensure_dspy_lm()):
                prediction = predictor(
                    context=context,
                    sensitive_entities=entities,
                    attack=_VECTOR_GUIDANCE[vector],
                )
        except Exception as e:
            logger.warning(
                "Leakage probe '%s' failed (%s); treating as no leak", vector.value, e
            )
            return LeakageVerdict(
                leaked=False,
                vector=vector.value,
                entity_type="NONE",
                evidence="",
                confidence=0.0,
            )
        confidence = _clamp_confidence(getattr(prediction, "confidence", 0.0))
        return LeakageVerdict(
            leaked=confidence >= self.leakage_threshold,
            vector=vector.value,
            entity_type=str(getattr(prediction, "entity_type", "NONE")).strip()
            or "NONE",
            evidence=str(getattr(prediction, "evidence", "")).strip(),
            confidence=confidence,
        )

    def probe(
        self, policy: PrivacyPolicy, sanitized_chunks: List[str]
    ) -> LeakageVerdict:
        """Attack the sanitized context and return the strongest leakage signal."""
        context = "\n\n".join(c for c in sanitized_chunks if c).strip()
        if not context or not self.strategies:
            return _SAFE_VERDICT

        predictor = _build_probe_predictor()
        entities = list(policy.sensitive_entities)
        verdicts = [
            self._probe_vector(predictor, context, entities, vector)
            for vector in self.strategies
        ]
        return max(verdicts, key=lambda v: v.confidence)

    def _node(self, state: PrivacyState) -> PrivacyState:
        """Graph node: write the leakage verdict and the safety flag to state."""
        policy = state.get("policy")
        if policy is None:
            raise ValueError("AdversarialAgent requires 'policy' in the state.")
        verdict = self.probe(policy, list(state.get("sanitized_chunks", [])))
        return PrivacyState(verdict=verdict, safe=not verdict.leaked)
