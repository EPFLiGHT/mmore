"""Context/Policy Analyzer.

Pipeline:  [analyzer]  ->  detector  ->  sanitizer  ->  adversarial -> ...
Reads:     query, raw_chunks
Writes:    policy

One node in the privacy multi-agent pipeline: picks the privacy domain
(explicit config or inferred from the query and raw chunks) and emits
the per-request PrivacyPolicy the next agents consume.
"""

import logging
import re
import secrets
from dataclasses import asdict, replace
from typing import Dict, List, Literal, Optional, Union

import dspy
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import Self

from ...rag.llm import LLMConfig
from ...utils import load_config
from ..config import PrivacyConfig
from ..detection.constants import (
    DETECTION_DEFAULT_PARAMS,
    DETECTION_GUIDANCE,
    DETECTION_PARAM_GUIDANCE,
    DETECTION_TOOL_NAMES,
    THRESHOLD_LEVELS,
)
from ..domains.profile import DOMAIN_PROFILES, get_domain_profile
from ..dspy_llm import TolerantJSONAdapter, build_dspy_lm
from ..escalation import apply_entity_guidance
from ..leakage import EscalationRecord
from ..policy import PrivacyPolicy
from ..sanitization.constants import (
    PRESIDIO_OPERATOR_GUIDANCE,
    SANITIZATION_GUIDANCE,
    SANITIZATION_TOOL_NAMES,
)
from ..ux import report_notice
from .base import BaseAgent
from .registry import tool_registry
from .state import PreCloudOutcome, PrivacyState

logger = logging.getLogger(__name__)


# ========================================================================
# Prompts
# ========================================================================

_DOMAIN_CLASSIFY_INSTRUCTION = (
    "Classify which privacy domain this retrieval-augmented request belongs "
    "to. Use 'healthcare' for clinical or medical content, 'humanitarian' "
    "for affected-population or displacement content, otherwise 'global'."
)

_ENGINE_SELECT_INSTRUCTION = (
    "Pick the single best PII detection engine for this request. Use the "
    "per-engine guidance to decide, and choose the engine whose strengths "
    "best match the request and context."
)

_PARAM_SELECT_INSTRUCTION = (
    "Pick the parameter values for the chosen detection engine. Use the "
    "engine-specific guidance to decide; default to the 'medium' threshold "
    "and the documented default for the other knobs unless the request or "
    "context clearly suggests otherwise."
)

_LABEL_EXPAND_INSTRUCTION = (
    "Propose any additional sensitive-entity labels that should be detected "
    "in this request and context, beyond the current set. Return them as "
    "uppercase identifiers like PASSPORT_NUMBER, BANK_ACCOUNT, BIOMETRIC_ID. "
    "Return an empty list if the current set already covers everything."
)

_FEEDBACK_LABEL_EXPAND_INSTRUCTION = (
    "A reviewer (the human at the gate or the leakage adversary) rejected the "
    "sanitized context and gave the feedback below. Propose additional "
    "sensitive-entity labels that act on that feedback, beyond the current "
    "set. Return them as uppercase identifiers like GPS_COORDINATES, "
    "RARE_DIAGNOSIS, JOB_TITLE. Return an empty list if the current set "
    "already covers the feedback."
)

_FEEDBACK_POLICY_INSTRUCTION = (
    "A reviewer (the human at the gate or the leakage adversary) gave the "
    "feedback below. Decide whether it calls for a different detection engine, "
    "sanitization strategy, detection threshold level, presidio anonymization "
    "operator, or a custom rewrite instruction, either by naming one "
    "explicitly or by describing what they want (use the guidance to map the "
    "description to the best fitting option). When the feedback describes an "
    "anonymization technique (masking characters, hashing, encryption), choose "
    "the presidio strategy and the matching operator. When it describes how "
    "the sanitized text should read, choose synthetic_rewrite and distill a "
    "short rewrite instruction. When it describes what or how to detect (e.g. "
    "treat certain terms or patterns as sensitive), choose the llm engine and "
    "distill a short detection instruction. Return the chosen value from the "
    "available options, 'keep' for a field the feedback does not affect, and "
    "'none' for no rewrite or detection instruction."
)


# ========================================================================
# DSPy signature field descriptions
# ========================================================================

_QUERY_DESC = "the user request"
_CONTEXT_DESC = "the retrieved context (concatenation of the raw chunks)"
_DOMAIN_DESC = "exactly one of: global, healthcare, humanitarian"
_DETECTION_GUIDANCE_DESC = "per-engine guidance: pros, cons, and when to prefer each"
_ENGINE_OUTPUT_DESC = "exactly one of: presidio, gliner, openai_filter, llm"
_PARAM_GUIDANCE_DESC = "engine-specific guidance for each tunable parameter"
_THRESHOLD_OUTPUT_DESC = "exactly one of: low, medium, high"
_CURRENT_ENTITIES_DESC = "the sensitive entity labels already in the policy"
_ADDITIONAL_ENTITIES_DESC = (
    "JSON array of uppercase identifier strings, e.g. "
    '["PASSPORT_NUMBER", "BANK_ACCOUNT"]. Empty array if nothing extra is needed.'
)
_FEEDBACK_DESC = (
    "the reviewer's free-text guidance: human gate feedback or the adversary's "
    "remediation report"
)
_STRATEGY_GUIDANCE_DESC = "per-strategy guidance: what each sanitization strategy does"
_AVAILABLE_ENGINES_DESC = "the detection engines you may choose from"
_AVAILABLE_STRATEGIES_DESC = "the sanitization strategies you may choose from"
_REQUESTED_ENGINE_DESC = "one of the available engines, or 'keep' to leave it unchanged"
_REQUESTED_STRATEGY_DESC = (
    "one of the available strategies, or 'keep' to leave it unchanged"
)
_REQUESTED_THRESHOLD_DESC = (
    "one of: low, medium, high, or 'keep' to leave the detection threshold unchanged"
)
_OPERATOR_GUIDANCE_DESC = (
    "per-operator guidance: what each presidio anonymization operator does"
)
_REQUESTED_OPERATOR_DESC = (
    "one of the presidio operators, or 'keep' to leave anonymization unchanged"
)
_REWRITE_INSTRUCTION_DESC = (
    "concise PII-free instruction for the rewrite LLM distilled from the "
    "feedback, or 'none'"
)
_DETECTION_INSTRUCTION_DESC = (
    "concise PII-free instruction for the LLM detector distilled from the "
    "feedback, or 'none'"
)

_MAX_ADDITIONAL_ENTITIES = 8
_LABEL_NON_ID_RE = re.compile(r"[^A-Z0-9_]")


# ========================================================================
# DSPy signatures
# ========================================================================


class _DomainClassifySignature(dspy.Signature):
    query: str = dspy.InputField(desc=_QUERY_DESC)
    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    domain: Literal["global", "healthcare", "humanitarian"] = dspy.OutputField(
        desc=_DOMAIN_DESC
    )


class _EngineSelectSignature(dspy.Signature):
    query: str = dspy.InputField(desc=_QUERY_DESC)
    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    engine_guidance: str = dspy.InputField(desc=_DETECTION_GUIDANCE_DESC)
    engine: Literal["presidio", "gliner", "openai_filter", "llm"] = dspy.OutputField(
        desc=_ENGINE_OUTPUT_DESC
    )


class _LabelExpandSignature(dspy.Signature):
    query: str = dspy.InputField(desc=_QUERY_DESC)
    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    current_entities: List[str] = dspy.InputField(desc=_CURRENT_ENTITIES_DESC)
    additional_entities: List[str] = dspy.OutputField(desc=_ADDITIONAL_ENTITIES_DESC)


class _FeedbackLabelExpandSignature(dspy.Signature):
    query: str = dspy.InputField(desc=_QUERY_DESC)
    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    current_entities: List[str] = dspy.InputField(desc=_CURRENT_ENTITIES_DESC)
    feedback: str = dspy.InputField(desc=_FEEDBACK_DESC)
    additional_entities: List[str] = dspy.OutputField(desc=_ADDITIONAL_ENTITIES_DESC)


class _FeedbackPolicySignature(dspy.Signature):
    feedback: str = dspy.InputField(desc=_FEEDBACK_DESC)
    engine_guidance: str = dspy.InputField(desc=_DETECTION_GUIDANCE_DESC)
    strategy_guidance: str = dspy.InputField(desc=_STRATEGY_GUIDANCE_DESC)
    operator_guidance: str = dspy.InputField(desc=_OPERATOR_GUIDANCE_DESC)
    available_engines: List[str] = dspy.InputField(desc=_AVAILABLE_ENGINES_DESC)
    available_strategies: List[str] = dspy.InputField(desc=_AVAILABLE_STRATEGIES_DESC)
    requested_engine: str = dspy.OutputField(desc=_REQUESTED_ENGINE_DESC)
    requested_strategy: str = dspy.OutputField(desc=_REQUESTED_STRATEGY_DESC)
    requested_threshold: str = dspy.OutputField(desc=_REQUESTED_THRESHOLD_DESC)
    requested_operator: str = dspy.OutputField(desc=_REQUESTED_OPERATOR_DESC)
    rewrite_instruction: str = dspy.OutputField(desc=_REWRITE_INSTRUCTION_DESC)
    detection_instruction: str = dspy.OutputField(desc=_DETECTION_INSTRUCTION_DESC)


class _PresidioParamsSignature(dspy.Signature):
    query: str = dspy.InputField(desc=_QUERY_DESC)
    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    param_guidance: str = dspy.InputField(desc=_PARAM_GUIDANCE_DESC)
    threshold_level: Literal["low", "medium", "high"] = dspy.OutputField(
        desc=_THRESHOLD_OUTPUT_DESC
    )


class _GLiNERParamsSignature(dspy.Signature):
    query: str = dspy.InputField(desc=_QUERY_DESC)
    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    param_guidance: str = dspy.InputField(desc=_PARAM_GUIDANCE_DESC)
    threshold_level: Literal["low", "medium", "high"] = dspy.OutputField(
        desc=_THRESHOLD_OUTPUT_DESC
    )
    multi_label: bool = dspy.OutputField(
        desc="true to allow overlapping label assignments on the same span"
    )


class _OpenAIFilterParamsSignature(dspy.Signature):
    query: str = dspy.InputField(desc=_QUERY_DESC)
    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    param_guidance: str = dspy.InputField(desc=_PARAM_GUIDANCE_DESC)
    threshold_level: Literal["low", "medium", "high"] = dspy.OutputField(
        desc=_THRESHOLD_OUTPUT_DESC
    )


class _LLMDetectionParamsSignature(dspy.Signature):
    query: str = dspy.InputField(desc=_QUERY_DESC)
    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    param_guidance: str = dspy.InputField(desc=_PARAM_GUIDANCE_DESC)
    threshold_level: Literal["low", "medium", "high"] = dspy.OutputField(
        desc=_THRESHOLD_OUTPUT_DESC
    )


_PARAM_SIGNATURES: Dict[str, type[dspy.Signature]] = {
    "presidio": _PresidioParamsSignature,
    "gliner": _GLiNERParamsSignature,
    "openai_filter": _OpenAIFilterParamsSignature,
    "llm": _LLMDetectionParamsSignature,
}


# ========================================================================
# Predictors and helpers
# ========================================================================


def _build_domain_predictor() -> dspy.Predict:
    return dspy.Predict(
        _DomainClassifySignature.with_instructions(_DOMAIN_CLASSIFY_INSTRUCTION)
    )


def _build_detection_engine_selector_predictor() -> dspy.Predict:
    return dspy.Predict(
        _EngineSelectSignature.with_instructions(_ENGINE_SELECT_INSTRUCTION)
    )


def _build_label_expansion_predictor() -> dspy.Predict:
    return dspy.Predict(
        _LabelExpandSignature.with_instructions(_LABEL_EXPAND_INSTRUCTION)
    )


def _build_feedback_label_expansion_predictor() -> dspy.Predict:
    return dspy.Predict(
        _FeedbackLabelExpandSignature.with_instructions(
            _FEEDBACK_LABEL_EXPAND_INSTRUCTION
        )
    )


def _build_feedback_policy_predictor() -> dspy.Predict:
    return dspy.Predict(
        _FeedbackPolicySignature.with_instructions(_FEEDBACK_POLICY_INSTRUCTION)
    )


def _sanitize_label_additions(raw: object, current: List[str]) -> List[str]:
    """Clean the LLM's proposed labels: uppercase, strip, drop empties and
    labels already in ``current``, cap at the configured maximum."""
    if not isinstance(raw, list):
        return []
    current_set = set(current)
    seen: set[str] = set()
    cleaned: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        label = _LABEL_NON_ID_RE.sub("", item.strip().upper())
        if not label or label in current_set or label in seen:
            continue
        seen.add(label)
        cleaned.append(label)
        if len(cleaned) >= _MAX_ADDITIONAL_ENTITIES:
            break
    return cleaned


def _build_param_predictor(engine: str) -> dspy.Predict | None:
    """Build the DSPy predictor for the engine's param signature, or None."""
    sig = _PARAM_SIGNATURES.get(engine)
    if sig is None:
        return None
    return dspy.Predict(sig.with_instructions(_PARAM_SELECT_INSTRUCTION))


def _describe_policy_changes(old: PrivacyPolicy, new: PrivacyPolicy) -> str:
    """Short, human-readable label for what the guidance actually changed."""
    changes: List[str] = []
    if new.detection_engine != old.detection_engine:
        changes.append(f"detecting with {new.detection_engine}")
    if new.sanitization_strategy != old.sanitization_strategy:
        changes.append(f"sanitizing with {new.sanitization_strategy}")
    old_threshold = old.detection_params.get("confidence_threshold")
    new_threshold = new.detection_params.get("confidence_threshold")
    if new_threshold != old_threshold:
        changes.append(f"detection threshold {new_threshold}")
    old_operator = old.sanitization_params.get("operator")
    new_operator = new.sanitization_params.get("operator")
    if new_operator != old_operator:
        changes.append(f"{new_operator} operator")
    if new.sanitizer_system_prompt != old.sanitizer_system_prompt:
        changes.append("new rewrite instruction")
    if new.detector_system_prompt != old.detector_system_prompt:
        changes.append("new detection instruction")
    added_labels = len(new.sensitive_entities) - len(old.sensitive_entities)
    if added_labels:
        changes.append(f"{added_labels} more entity labels")
    return ", ".join(changes) if changes else "same policy, stricter guidance"


def _format_engine_guidance() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in DETECTION_GUIDANCE.items())


def _format_strategy_guidance() -> str:
    return "\n".join(
        f"- {name}: {desc}" for name, desc in SANITIZATION_GUIDANCE.items()
    )


def _format_operator_guidance() -> str:
    return "\n".join(
        f"- {name}: {desc}" for name, desc in PRESIDIO_OPERATOR_GUIDANCE.items()
    )


def _detection_unchanged(old: PrivacyPolicy, new: PrivacyPolicy) -> bool:
    """True when the escalation left every detection-relevant field untouched."""
    return (
        new.detection_engine == old.detection_engine
        and new.detection_params == old.detection_params
        and new.detector_system_prompt == old.detector_system_prompt
        and set(new.sensitive_entities) == set(old.sensitive_entities)
    )


def _pin_rewrite_instruction(prompt: str, instruction: str) -> str:
    """Append the reviewer's rewrite instruction to the prompt, idempotently."""
    if not instruction or instruction.lower() in ("none", "keep"):
        return prompt
    note = f"Reviewer instruction: {instruction}"
    if note in prompt:
        return prompt
    return f"{prompt} {note}".strip()


def _parse_param_prediction(
    engine: str, prediction: dspy.Prediction
) -> Dict[str, Union[float, bool]] | None:
    """Parse and validate the generated engine parameters."""
    params: Dict[str, Union[float, bool]] = {}
    raw_level = str(getattr(prediction, "threshold_level", "")).strip().lower()
    if raw_level not in THRESHOLD_LEVELS:
        return None
    params["confidence_threshold"] = THRESHOLD_LEVELS[raw_level]
    if engine == "gliner":
        value = getattr(prediction, "multi_label", None)
        if not isinstance(value, bool):
            return None
        params["multi_label"] = value
    return params


# ========================================================================
# Agent
# ========================================================================


class ContextPolicyAnalyzerAgent(BaseAgent):
    """Selects the domain and emits the per-request privacy policy."""

    state_schema = PrivacyState
    node_name = "analyzer"

    def __init__(
        self,
        config: PrivacyConfig,
        llm_config: Optional[LLMConfig] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self._dspy_lm: Optional[dspy.BaseLM] = None
        super().__init__(config, llm_config=llm_config, checkpointer=checkpointer)

    @classmethod
    def from_config(
        cls,
        config: PrivacyConfig | str,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> Self:
        if not isinstance(config, PrivacyConfig):
            config = load_config(config, PrivacyConfig)
        llm_config = config.context_analyzer.llm if config.context_analyzer else None
        return cls(config, llm_config, checkpointer=checkpointer)

    def _ensure_dspy_lm(self) -> dspy.BaseLM:
        """Lazily build the DSPy LM, raising an error if no LLM is configured."""
        if self._dspy_lm is None:
            if self._llm_config is None:
                raise ValueError(
                    "Analyzer agent requires an LLM to predict privacy policy parameters."
                )
            self._dspy_lm = build_dspy_lm(self._llm_config)
        return self._dspy_lm

    def _infer_domain(self, query: str, chunks: List[str]) -> str:
        self._ensure_dspy_lm()
        predictor = _build_domain_predictor()
        try:
            with dspy.context(lm=self._dspy_lm):
                prediction = predictor(query=query, context="\n\n".join(chunks))
        except Exception as e:
            logger.warning("Domain inference failed (%s); defaulting to 'global'", e)
            return "global"
        domain = str(getattr(prediction, "domain", "")).strip().lower()
        return domain if domain in DOMAIN_PROFILES else "global"

    def _select_engine(self, query: str, chunks: List[str]) -> str | None:
        """Pick a detection engine via DSPy."""
        self._ensure_dspy_lm()
        predictor = _build_detection_engine_selector_predictor()
        try:
            with dspy.context(lm=self._dspy_lm):
                prediction = predictor(
                    query=query,
                    context="\n\n".join(chunks),
                    engine_guidance=_format_engine_guidance(),
                )
        except Exception as e:
            logger.warning(
                "Engine selection failed (%s), falling back to profile defaults",
                e,
            )
            return None
        engine = str(getattr(prediction, "engine", "")).strip().lower()
        return (
            engine
            if engine in DETECTION_TOOL_NAMES
            and DETECTION_TOOL_NAMES[engine] in tool_registry
            else None
        )

    def _resolve_engine(self, requested: str, profile_default: str) -> str:
        """Validate the requested engine."""
        if (
            requested in DETECTION_TOOL_NAMES
            and DETECTION_TOOL_NAMES[requested] in tool_registry
        ):
            return requested
        logger.warning(
            "The given engine (%s) cannot be resolved, falling back to default (%s)",
            requested,
            profile_default,
        )
        return profile_default

    def _expand_labels(
        self, query: str, chunks: List[str], current: List[str]
    ) -> List[str]:
        """Propose extra sensitive-entity labels via DSPy."""
        self._ensure_dspy_lm()
        predictor = _build_label_expansion_predictor()
        try:
            with dspy.context(lm=self._dspy_lm, adapter=TolerantJSONAdapter()):
                prediction = predictor(
                    query=query,
                    context="\n\n".join(chunks),
                    current_entities=list(current),
                )
        except Exception as e:
            logger.warning(
                "Label expansion failed (%s), keeping the current entity set", e
            )
            return []
        return _sanitize_label_additions(
            getattr(prediction, "additional_entities", None), current
        )

    def _expand_labels_from_feedback(
        self, state: PrivacyState, policy: PrivacyPolicy, feedback: str
    ) -> List[str]:
        """Propose sensitive labels that act on the reviewer's guidance."""
        if self._llm_config is None or not feedback:
            return []
        current = list(policy.sensitive_entities)
        self._ensure_dspy_lm()
        predictor = _build_feedback_label_expansion_predictor()
        try:
            with dspy.context(lm=self._dspy_lm, adapter=TolerantJSONAdapter()):
                prediction = predictor(
                    query=state.get("query", ""),
                    context="\n\n".join(state.get("raw_chunks", [])),
                    current_entities=current,
                    feedback=feedback,
                )
        except Exception as e:
            logger.warning("Feedback-driven label expansion failed (%s)", e)
            return []
        return _sanitize_label_additions(
            getattr(prediction, "additional_entities", None), current
        )

    def _apply_feedback_overrides(
        self, policy: PrivacyPolicy, feedback: str, respect_config_pins: bool
    ) -> PrivacyPolicy:
        """Switch engine/strategy/threshold per the guidance; pins bind the adversary only."""
        if self._llm_config is None or not feedback:
            return policy
        self._ensure_dspy_lm()
        predictor = _build_feedback_policy_predictor()
        try:
            with dspy.context(lm=self._dspy_lm):
                prediction = predictor(
                    feedback=feedback,
                    engine_guidance=_format_engine_guidance(),
                    strategy_guidance=_format_strategy_guidance(),
                    operator_guidance=_format_operator_guidance(),
                    available_engines=list(DETECTION_TOOL_NAMES),
                    available_strategies=list(SANITIZATION_TOOL_NAMES),
                )
        except Exception as e:
            logger.warning("Feedback policy override failed (%s)", e)
            return policy
        engine = str(getattr(prediction, "requested_engine", "")).strip().lower()
        strategy = str(getattr(prediction, "requested_strategy", "")).strip().lower()
        threshold = str(getattr(prediction, "requested_threshold", "")).strip().lower()
        operator = str(getattr(prediction, "requested_operator", "")).strip().lower()
        instruction = str(getattr(prediction, "rewrite_instruction", "")).strip()
        detection_instruction = str(
            getattr(prediction, "detection_instruction", "")
        ).strip()
        engine_pinned = respect_config_pins and self.config.detection.engine
        strategy_pinned = respect_config_pins and self.config.sanitization.strategy
        threshold_pinned = (
            respect_config_pins
            and self.config.detection.confidence_threshold is not None
        )
        updates: Dict[str, object] = {}
        params = dict(policy.detection_params)
        params_changed = False
        if engine in DETECTION_TOOL_NAMES and not engine_pinned:
            updates["detection_engine"] = engine
            if engine != policy.detection_engine:
                current_threshold = params.get("confidence_threshold")
                params = asdict(DETECTION_DEFAULT_PARAMS[engine])
                if current_threshold is not None:
                    params["confidence_threshold"] = current_threshold
                params_changed = True
        if strategy in SANITIZATION_TOOL_NAMES and not strategy_pinned:
            updates["sanitization_strategy"] = strategy
        if threshold in THRESHOLD_LEVELS and not threshold_pinned:
            params["confidence_threshold"] = THRESHOLD_LEVELS[threshold]
            params_changed = True
        if params_changed:
            updates["detection_params"] = params
        # Operator and rewrite instruction only apply under their own strategy
        final = updates.get("sanitization_strategy", policy.sanitization_strategy)
        if final == "presidio" and operator in PRESIDIO_OPERATOR_GUIDANCE:
            updates["sanitization_params"] = {
                "operator": operator,
                "operator_params": self._operator_params(operator),
            }
        if final == "synthetic_rewrite":
            prompt = _pin_rewrite_instruction(
                policy.sanitizer_system_prompt, instruction
            )
            if prompt != policy.sanitizer_system_prompt:
                updates["sanitizer_system_prompt"] = prompt
        final_engine = updates.get("detection_engine", policy.detection_engine)
        if final_engine == "llm":
            detector_prompt = _pin_rewrite_instruction(
                policy.detector_system_prompt, detection_instruction
            )
            if detector_prompt != policy.detector_system_prompt:
                updates["detector_system_prompt"] = detector_prompt
        return replace(policy, **updates) if updates else policy

    def _operator_params(self, operator: str) -> Dict[str, object]:
        """Default params for a presidio operator."""
        if operator == "mask":
            return {"masking_char": "*", "chars_to_mask": 128, "from_end": False}
        if operator == "encrypt":
            key = self.config.sanitization.encryption_key
            if not key:
                key = secrets.token_hex(16)
                logger.warning(
                    "No sanitization.encryption_key configured: using an "
                    "ephemeral key, encrypted values are unrecoverable."
                )
            return {"key": key}
        return {}

    def _select_params(
        self, engine: str, query: str, chunks: List[str]
    ) -> Dict[str, Union[float, bool]] | None:
        """Pick ``engine``'s tunable params via DSPy."""
        if self._llm_config is None:
            return None
        predictor = _build_param_predictor(engine)
        if predictor is None:
            return None
        self._ensure_dspy_lm()
        try:
            with dspy.context(lm=self._dspy_lm):
                prediction = predictor(
                    query=query,
                    context="\n\n".join(chunks),
                    param_guidance=DETECTION_PARAM_GUIDANCE.get(engine, ""),
                )
        except Exception as e:
            logger.warning(
                "Param selection failed for %s (%s), falling back to engine defaults",
                engine,
                e,
            )
            return None
        return _parse_param_prediction(engine, prediction)

    def build_policy(self, query: str, chunks: List[str]) -> PrivacyPolicy:
        """Resolve the domain and merge profile defaults with config overrides."""
        domain = self.config.domain or self._infer_domain(query, chunks)
        profile = get_domain_profile(domain)

        detection_cfg = self.config.detection
        sanitization_cfg = self.config.sanitization

        if detection_cfg.engine:
            engine_short = self._resolve_engine(
                detection_cfg.engine, profile.default_engine
            )
        else:
            engine_short = self._select_engine(query, chunks) or profile.default_engine

        defaults = DETECTION_DEFAULT_PARAMS[engine_short]
        if detection_cfg.confidence_threshold is not None:
            detection_params = asdict(defaults)
            detection_params["confidence_threshold"] = (
                detection_cfg.confidence_threshold
            )
        else:
            selected = self._select_params(engine_short, query, chunks)
            detection_params = selected if selected is not None else asdict(defaults)

        strategy = sanitization_cfg.strategy or profile.default_strategy
        consistency = (
            sanitization_cfg.consistency
            if sanitization_cfg.consistency is not None
            else profile.default_consistency
        )
        if detection_cfg.entity_types:
            sensitive_entities = list(detection_cfg.entity_types)
        else:
            current_entities = list(profile.sensitive_entities)
            sensitive_entities = current_entities + self._expand_labels(
                query, chunks, current_entities
            )

        return PrivacyPolicy(
            domain=domain,
            sensitive_entities=list(sensitive_entities),
            detection_engine=engine_short,
            detection_params=detection_params,
            sanitization_strategy=strategy,
            consistency=consistency,
            domain_prompt=profile.domain_prompt,
            sanitizer_system_prompt=profile.sanitizer_system_prompt,
        )

    def _apply_guidance(
        self,
        state: PrivacyState,
        policy: PrivacyPolicy,
        guidance: str,
        note_label: str,
        respect_config_pins: bool,
    ) -> PrivacyPolicy:
        """Act on a reviewer's guidance text; note-only hardening without one."""
        if not guidance:
            return apply_entity_guidance(policy)
        extra_entities = self._expand_labels_from_feedback(state, policy, guidance)
        new_policy = apply_entity_guidance(
            policy, extra_entities or None, guidance, note_label=note_label
        )
        return self._apply_feedback_overrides(new_policy, guidance, respect_config_pins)

    def _escalate(self, state: PrivacyState, policy: PrivacyPolicy) -> PrivacyState:
        """Re-entry path: harden the policy from the adversary report or gate feedback."""
        total_escalations = state.get("total_escalations", 0)
        adversary_escalations = state.get("adversary_escalations", 0)
        verdict = state.get("verdict")
        trigger_vector, trigger_entity = None, None
        from_human_feedback = False
        if state.get("revision_requested"):
            revision = total_escalations - adversary_escalations + 1
            reason = f"Revision {revision}: acting on your feedback"
            report = (state.get("human_feedback") or "").strip()
            new_policy = self._apply_guidance(
                state,
                policy,
                report,
                note_label="Human guidance",
                respect_config_pins=False,
            )
            from_human_feedback = bool(report)
        elif verdict is not None and verdict.leaked:
            report = (verdict.recommendation or "").strip()
            new_policy = self._apply_guidance(
                state,
                policy,
                report,
                note_label="Adversary guidance",
                respect_config_pins=True,
            )
            trigger_vector, trigger_entity = verdict.vector, verdict.entity_type
            adversary_escalations += 1
            budget = self.config.leakage_adversary.max_iterations
            leaked = trigger_entity or "sensitive data"
            vector = (
                trigger_vector.value.replace("_", " ") if trigger_vector else "a probe"
            )
            reason = (
                f"Leak {adversary_escalations}/{budget}: the adversary recovered "
                f"{leaked} via {vector} (confidence {verdict.confidence:.2f})"
            )
        else:
            raise ValueError("escalate called without a leak or a revision")
        label = _describe_policy_changes(policy, new_policy)
        record = EscalationRecord(
            iteration=total_escalations + 1,
            escalation=label,
            from_human_feedback=from_human_feedback,
            vector=trigger_vector,
            entity_type=trigger_entity,
            report=report or None,
        )
        notice = f"{reason} → retrying with {label}"
        if not report_notice(notice):
            logger.info("Privacy: %s", notice)
        return PrivacyState(
            policy=new_policy,
            total_escalations=total_escalations + 1,
            adversary_escalations=adversary_escalations,
            escalation_log=list(state.get("escalation_log", [])) + [record],
            outcome=PreCloudOutcome.RE_LOOPED,
            human_feedback=None,  # was taken into account already
            revision_requested=False,
            skip_detection=_detection_unchanged(policy, new_policy),
        )

    def _node(self, state: PrivacyState) -> PrivacyState:
        """Graph node: build the policy on first entry, escalate it on re-entry.

        A policy already in the state means the loop or the gate sent the
        request back for a stricter pass, so the analyzer escalates rather than
        rebuilding from scratch.
        """
        policy = state.get("policy")
        if policy is None:
            policy = self.build_policy(
                state.get("query", ""), list(state.get("raw_chunks", []))
            )
            return PrivacyState(policy=policy)
        return self._escalate(state, policy)
