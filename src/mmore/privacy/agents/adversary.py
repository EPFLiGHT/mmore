"""Pre-cloud Leakage Adversary.

Pipeline:  analyzer -> detector -> sanitizer -> [leakage_adversary] -> gate
Reads:     policy, sanitized_chunks
Writes:    verdict, safe

The trust-boundary probe: it attacks the sanitized context for residual PII and
quasi-identifiers before anything leaves for the cloud answer model. It runs one
adversarial probe per configured attack vector, keeps the strongest signal, and
treats a probe whose confidence reaches the threshold as a leak. On a leak it
also emits a PII-free remediation report the analyzer applies like human gate
feedback: which engine, strategy, threshold, or entity labels to change.
"""

import logging
from dataclasses import replace
from typing import Dict, List, Optional

import dspy
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import Self

from ...rag.llm import LLMConfig
from ...utils import load_config
from ..config import AttackVector, PrivacyConfig
from ..detection.constants import (
    DEFAULT_LLM_CONFIG,
    DETECTION_GUIDANCE,
    DETECTION_PARAM_GUIDANCE,
)
from ..dspy_llm import build_dspy_lm
from ..leakage import SAFE_VERDICT, LeakageVerdict
from ..policy import PrivacyPolicy
from ..sanitization.constants import (
    PRESIDIO_OPERATOR_GUIDANCE,
    SANITIZATION_GUIDANCE,
)
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
    "can actually be recovered or confidently inferred. Report a leak only "
    "when a specific individual's identifier genuinely remains recoverable; "
    "masked, generic, or non-identifying content is not a leak. Calibrate the "
    "confidence to the true strength of the residual signal. Do not include raw "
    "personal values in your evidence."
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


# ========================================================================
# Remediation report (emitted on a leak, applied by the analyzer)
# ========================================================================

_CURRENT_POLICY_DESC = (
    "the policy the leaking pass ran under: detection engine, sanitization "
    "strategy, confidence threshold, and sensitive-entity labels"
)
_ENGINE_GUIDANCE_DESC = "per-engine guidance: pros, cons, and when to prefer each"
_STRATEGY_GUIDANCE_DESC = "per-strategy guidance: what each sanitization strategy does"
_OPERATOR_GUIDANCE_DESC = (
    "per-operator guidance: what each presidio anonymization operator does"
)
_PARAM_GUIDANCE_DESC = "per-engine guidance for each tunable parameter"
_FIXED_FIELDS_DESC = (
    "policy fields pinned by the user config that must not be changed, or 'none'"
)
_PREVIOUS_REPORTS_DESC = "your remediation reports from earlier iterations, or 'none'"
_RECOMMENDATION_DESC = (
    "concise PII-free remediation report for the next pass: which detection "
    "engine, sanitization strategy, presidio anonymization operator, or "
    "threshold level (low/medium/high) to use, how the rewritten text should "
    "read, and any sensitive-entity labels to add"
)

_REMEDIATION_INSTRUCTION = (
    "You are a privacy red-team adversary reporting back to the policy analyzer "
    "after a successful attack on the sanitized context. Using the tool guidance, "
    "write a short remediation report for the next pass: recommend a detection "
    "engine, sanitization strategy, presidio anonymization operator, threshold "
    "level, rewrite instruction, and/or additional sensitive-entity labels that "
    "would close the leak. Never recommend changing a field listed as fixed by "
    "the user. Compare the current policy against your previous reports: if an "
    "earlier recommendation was not applied, say so and restate it. Do not echo "
    "raw personal values."
)


class _RemediationSignature(dspy.Signature):
    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    attack: str = dspy.InputField(desc="the attack that succeeded and how it works")
    entity_type: str = dspy.InputField(desc="the entity type that leaked, or NONE")
    evidence: str = dspy.InputField(desc="the probe's PII-free justification")
    current_policy: str = dspy.InputField(desc=_CURRENT_POLICY_DESC)
    engine_guidance: str = dspy.InputField(desc=_ENGINE_GUIDANCE_DESC)
    strategy_guidance: str = dspy.InputField(desc=_STRATEGY_GUIDANCE_DESC)
    operator_guidance: str = dspy.InputField(desc=_OPERATOR_GUIDANCE_DESC)
    param_guidance: str = dspy.InputField(desc=_PARAM_GUIDANCE_DESC)
    fixed_fields: str = dspy.InputField(desc=_FIXED_FIELDS_DESC)
    previous_reports: str = dspy.InputField(desc=_PREVIOUS_REPORTS_DESC)
    recommendation: str = dspy.OutputField(desc=_RECOMMENDATION_DESC)


def _build_remediation_predictor() -> dspy.Predict:
    return dspy.Predict(
        _RemediationSignature.with_instructions(_REMEDIATION_INSTRUCTION)
    )


def _describe_policy(policy: PrivacyPolicy) -> str:
    threshold = policy.detection_params.get("confidence_threshold", "default")
    return (
        f"engine={policy.detection_engine}, "
        f"strategy={policy.sanitization_strategy}, "
        f"threshold={threshold}, "
        f"entities={', '.join(policy.sensitive_entities) or 'none'}"
    )


def _format_guidance(guidance: Dict[str, str]) -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in guidance.items())


def _clamp_confidence(value: float | int | str | None) -> float:
    """Coerce a model-provided confidence into ``[0.0, 1.0]``, 0.0 on failure."""
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    try:
        return max(0.0, min(1.0, float(str(value))))
    except (TypeError, ValueError):
        return 0.0


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
        if self._adversary_cfg.enabled and llm_config is None:
            llm_config = (
                config.context_analyzer.llm
                if config.context_analyzer
                else DEFAULT_LLM_CONFIG
            )
            logger.warning(
                "No leakage_adversary.llm configured, falling back to %s",
                llm_config.llm_name,
            )
        super().__init__(config, llm_config=llm_config, checkpointer=checkpointer)

    @classmethod
    def from_config(
        cls,
        config: PrivacyConfig | str,
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
        """Run one attack vector, if we get an error it's considered as a leak."""
        try:
            with dspy.context(lm=self._ensure_dspy_lm()):
                prediction = predictor(
                    context=context,
                    sensitive_entities=entities,
                    attack=_VECTOR_GUIDANCE[vector],
                )
        except Exception as e:
            logger.warning(
                "Leakage probe '%s' failed (%s), treating as a leak", vector.value, e
            )
            return LeakageVerdict(
                leaked=True,
                vector=vector,
                entity_type=None,
                evidence="probe_failed",
                confidence=1.0,
            )
        confidence = _clamp_confidence(getattr(prediction, "confidence", 0.0))
        entity = str(getattr(prediction, "entity_type", "")).strip()
        return LeakageVerdict(
            leaked=confidence >= self.leakage_threshold,
            vector=vector,
            entity_type=entity if entity and entity.upper() != "NONE" else None,
            evidence=str(getattr(prediction, "evidence", "")).strip(),
            confidence=confidence,
        )

    def probe(
        self, policy: PrivacyPolicy, sanitized_chunks: List[str]
    ) -> LeakageVerdict:
        """Attack the sanitized context and return the strongest leakage signal."""
        context = "\n\n".join(c for c in sanitized_chunks if c).strip()
        if not context or not self.strategies:
            return SAFE_VERDICT

        predictor = _build_probe_predictor()
        entities = list(policy.sensitive_entities)
        verdicts = []
        for number, vector in enumerate(self.strategies, 1):
            logger.debug(
                "Adversary probe (LLM): %s (%d/%d)",
                vector.value,
                number,
                len(self.strategies),
            )
            verdicts.append(self._probe_vector(predictor, context, entities, vector))
        return max(verdicts, key=lambda v: v.confidence)

    def _fixed_policy_fields(self) -> str:
        """List the policy fields the user pinned in the config."""
        fixed = []
        if self.config.detection.engine:
            fixed.append(f"detection engine ({self.config.detection.engine.value})")
        if self.config.detection.confidence_threshold is not None:
            fixed.append(
                f"confidence threshold ({self.config.detection.confidence_threshold})"
            )
        if self.config.sanitization.strategy:
            fixed.append(
                f"sanitization strategy ({self.config.sanitization.strategy.value})"
            )
        return "; ".join(fixed) or "none"

    def recommend(
        self,
        policy: PrivacyPolicy,
        context: str,
        verdict: LeakageVerdict,
        previous_reports: List[str],
    ) -> Optional[str]:
        """Write the remediation report for a leaking verdict, or None on failure."""
        predictor = _build_remediation_predictor()
        try:
            with dspy.context(lm=self._ensure_dspy_lm()):
                prediction = predictor(
                    context=context,
                    attack=_VECTOR_GUIDANCE[verdict.vector] if verdict.vector else "",
                    entity_type=verdict.entity_type or "NONE",
                    evidence=verdict.evidence,
                    current_policy=_describe_policy(policy),
                    engine_guidance=_format_guidance(DETECTION_GUIDANCE),
                    strategy_guidance=_format_guidance(SANITIZATION_GUIDANCE),
                    operator_guidance=_format_guidance(PRESIDIO_OPERATOR_GUIDANCE),
                    param_guidance=_format_guidance(DETECTION_PARAM_GUIDANCE),
                    fixed_fields=self._fixed_policy_fields(),
                    previous_reports="\n---\n".join(previous_reports) or "none",
                )
        except Exception as e:
            logger.warning("Remediation report failed (%s), escalating without one", e)
            return None
        report = str(getattr(prediction, "recommendation", "")).strip()
        return report or None

    def _node(self, state: PrivacyState) -> PrivacyState:
        """Graph node: write the verdict (with its report on a leak) and the safety flag."""
        if not self._adversary_cfg.enabled:
            return PrivacyState(verdict=SAFE_VERDICT, safe=True)
        policy = state.get("policy")
        if policy is None:
            raise ValueError("AdversarialAgent requires 'policy' in the state.")
        chunks = list(state.get("sanitized_chunks", []))
        verdict = self.probe(policy, chunks)
        if verdict.leaked:
            previous = [
                r.report
                for r in state.get("escalation_log", [])
                if r.report and not r.from_human_feedback
            ]
            context = "\n\n".join(c for c in chunks if c).strip()
            report = self.recommend(policy, context, verdict, previous)
            verdict = replace(verdict, recommendation=report)
        return PrivacyState(verdict=verdict, safe=not verdict.leaked)
