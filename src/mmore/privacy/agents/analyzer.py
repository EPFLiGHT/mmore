"""Context/Policy Analyzer.

Pipeline:  [analyzer]  ->  detector  ->  sanitizer  ->  adversarial -> ...
Reads:     query, raw_chunks
Writes:    policy

One node in the privacy multi-agent pipeline: picks the privacy domain
(explicit config or inferred from the query and raw chunks) and emits
the per-request PrivacyPolicy the next agents consume. On re-entry it is also
the single policy authority: it hardens the policy from the adversary's
remediation report or from the human's gate feedback.
"""

import logging
import re
import secrets
from dataclasses import asdict, replace
from typing import Literal

import dspy
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import Self

from ...rag.llm import LLMConfig
from ..config import PrivacyConfig, as_privacy_config
from ..detection.constants import (
    DETECTION_DEFAULT_PARAMS,
    DETECTION_GUIDANCE,
    DETECTION_PARAM_GUIDANCE,
    DETECTION_TOOL_NAMES,
    THRESHOLD_LEVELS,
)
from ..domains import DOMAIN_PROFILES, get_domain_profile
from ..dspy_llm import TolerantJSONAdapter, format_guidance
from ..sanitization.constants import (
    PRESIDIO_OPERATOR_GUIDANCE,
    SANITIZATION_GUIDANCE,
    SANITIZATION_TOOL_NAMES,
)
from ..schemas.leakage import EscalationRecord
from ..schemas.policy import PrivacyPolicy, harden
from ..schemas.report import PreCloudOutcome
from ..ux import report_notice
from .base import BaseAgent
from .registry import tool_registry
from .state import PrivacyState

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

# Values the feedback predictor returns to mean "do not touch this field"
_UNCHANGED = ("keep", "none", "")


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
    current_entities: list[str] = dspy.InputField(desc=_CURRENT_ENTITIES_DESC)
    additional_entities: list[str] = dspy.OutputField(desc=_ADDITIONAL_ENTITIES_DESC)


class _FeedbackLabelExpandSignature(dspy.Signature):
    query: str = dspy.InputField(desc=_QUERY_DESC)
    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    current_entities: list[str] = dspy.InputField(desc=_CURRENT_ENTITIES_DESC)
    feedback: str = dspy.InputField(desc=_FEEDBACK_DESC)
    additional_entities: list[str] = dspy.OutputField(desc=_ADDITIONAL_ENTITIES_DESC)


class _FeedbackPolicySignature(dspy.Signature):
    feedback: str = dspy.InputField(desc=_FEEDBACK_DESC)
    engine_guidance: str = dspy.InputField(desc=_DETECTION_GUIDANCE_DESC)
    strategy_guidance: str = dspy.InputField(desc=_STRATEGY_GUIDANCE_DESC)
    operator_guidance: str = dspy.InputField(desc=_OPERATOR_GUIDANCE_DESC)
    available_engines: list[str] = dspy.InputField(desc=_AVAILABLE_ENGINES_DESC)
    available_strategies: list[str] = dspy.InputField(desc=_AVAILABLE_STRATEGIES_DESC)
    requested_engine: str = dspy.OutputField(desc=_REQUESTED_ENGINE_DESC)
    requested_strategy: str = dspy.OutputField(desc=_REQUESTED_STRATEGY_DESC)
    requested_threshold: str = dspy.OutputField(desc=_REQUESTED_THRESHOLD_DESC)
    requested_operator: str = dspy.OutputField(desc=_REQUESTED_OPERATOR_DESC)
    rewrite_instruction: str = dspy.OutputField(desc=_REWRITE_INSTRUCTION_DESC)
    detection_instruction: str = dspy.OutputField(desc=_DETECTION_INSTRUCTION_DESC)


class _ThresholdParamsSignature(dspy.Signature):
    query: str = dspy.InputField(desc=_QUERY_DESC)
    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    param_guidance: str = dspy.InputField(desc=_PARAM_GUIDANCE_DESC)
    threshold_level: Literal["low", "medium", "high"] = dspy.OutputField(
        desc=_THRESHOLD_OUTPUT_DESC
    )


class _GLiNERParamsSignature(_ThresholdParamsSignature):
    multi_label: bool = dspy.OutputField(
        desc="true to allow overlapping label assignments on the same span"
    )


# Engines other than GLiNER expose the threshold only
_PARAM_SIGNATURES: dict[str, type[dspy.Signature]] = {
    "presidio": _ThresholdParamsSignature,
    "gliner": _GLiNERParamsSignature,
    "openai_filter": _ThresholdParamsSignature,
    "llm": _ThresholdParamsSignature,
}


# ========================================================================
# Helpers
# ========================================================================


def _predictor(signature: type[dspy.Signature], instruction: str) -> dspy.Predict:
    return dspy.Predict(signature.with_instructions(instruction))


def _read(prediction: dspy.Prediction, field: str) -> str:
    """Read one output field as a normalized lowercase string."""
    return str(getattr(prediction, field, "")).strip().lower()


def _clean_label_additions(raw: object, current: list[str]) -> list[str]:
    """Clean the LLM's proposed labels: uppercase, strip, drop empties and
    labels already in ``current``, cap at the configured maximum."""
    if not isinstance(raw, list):
        return []
    known = set(current)
    cleaned: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        label = _LABEL_NON_ID_RE.sub("", item.strip().upper())
        if not label or label in known:
            continue
        known.add(label)
        cleaned.append(label)
        if len(cleaned) >= _MAX_ADDITIONAL_ENTITIES:
            break
    return cleaned


def _describe_policy_changes(old: PrivacyPolicy, new: PrivacyPolicy) -> str:
    """Short, human-readable label for what the guidance actually changed."""
    changes: list[str] = []
    if new.detection_engine != old.detection_engine:
        changes.append(f"detecting with {new.detection_engine}")
    if new.sanitization_strategy != old.sanitization_strategy:
        changes.append(f"sanitizing with {new.sanitization_strategy}")
    new_threshold = new.detection_params.get("confidence_threshold")
    if new_threshold != old.detection_params.get("confidence_threshold"):
        changes.append(f"detection threshold {new_threshold}")
    new_operator = new.sanitization_params.get("operator")
    if new_operator != old.sanitization_params.get("operator"):
        changes.append(f"{new_operator} operator")
    if new.sanitizer_system_prompt != old.sanitizer_system_prompt:
        changes.append("new rewrite instruction")
    if new.detector_system_prompt != old.detector_system_prompt:
        changes.append("new detection instruction")
    added_labels = len(new.sensitive_entities) - len(old.sensitive_entities)
    if added_labels:
        changes.append(f"{added_labels} more entity labels")
    return ", ".join(changes) if changes else "same policy, stricter guidance"


def _detection_unchanged(old: PrivacyPolicy, new: PrivacyPolicy) -> bool:
    """True when the escalation left every detection-relevant field untouched."""
    return (
        new.detection_engine == old.detection_engine
        and new.detection_params == old.detection_params
        and new.detector_system_prompt == old.detector_system_prompt
        and set(new.sensitive_entities) == set(old.sensitive_entities)
    )


def _pin_instruction(prompt: str, instruction: str) -> str:
    """Append the reviewer's instruction to the prompt, idempotently."""
    if not instruction or instruction.lower() in _UNCHANGED:
        return prompt
    note = f"Reviewer instruction: {instruction}"
    return prompt if note in prompt else f"{prompt} {note}".strip()


def _parse_params(engine: str, prediction: dspy.Prediction) -> dict | None:
    """Parse and validate the generated engine parameters."""
    level = _read(prediction, "threshold_level")
    if level not in THRESHOLD_LEVELS:
        return None
    params: dict[str, float | bool] = {"confidence_threshold": THRESHOLD_LEVELS[level]}
    if engine == "gliner":
        multi_label = getattr(prediction, "multi_label", None)
        if not isinstance(multi_label, bool):
            return None
        params["multi_label"] = multi_label
    return params


def _engine_is_available(engine: str) -> bool:
    return (
        engine in DETECTION_TOOL_NAMES and DETECTION_TOOL_NAMES[engine] in tool_registry
    )


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
        llm_config: LLMConfig | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        super().__init__(config, llm_config=llm_config, checkpointer=checkpointer)

    @classmethod
    def from_config(
        cls,
        config: PrivacyConfig | str,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> Self:
        config = as_privacy_config(config)
        llm_config = config.context_analyzer.llm if config.context_analyzer else None
        return cls(config, llm_config, checkpointer=checkpointer)

    # -- DSPy calls ------------------------------------------------------

    def _run(
        self,
        lm: dspy.BaseLM,
        signature: type[dspy.Signature],
        instruction: str,
        adapter: dspy.Adapter | None = None,
        **inputs,
    ) -> dspy.Prediction:
        with dspy.context(lm=lm, **({"adapter": adapter} if adapter else {})):
            return _predictor(signature, instruction)(**inputs)

    def _infer_domain(self, query: str, chunks: list[str]) -> str:
        lm = self.dspy_lm
        try:
            prediction = self._run(
                lm,
                _DomainClassifySignature,
                _DOMAIN_CLASSIFY_INSTRUCTION,
                query=query,
                context="\n\n".join(chunks),
            )
        except Exception as e:
            logger.warning("Domain inference failed (%s); defaulting to 'global'", e)
            return "global"
        domain = _read(prediction, "domain")
        return domain if domain in DOMAIN_PROFILES else "global"

    def _select_engine(self, query: str, chunks: list[str]) -> str | None:
        """Pick a detection engine via DSPy, or None to keep the profile default."""
        lm = self.dspy_lm
        try:
            prediction = self._run(
                lm,
                _EngineSelectSignature,
                _ENGINE_SELECT_INSTRUCTION,
                query=query,
                context="\n\n".join(chunks),
                engine_guidance=format_guidance(DETECTION_GUIDANCE),
            )
        except Exception as e:
            logger.warning(
                "Engine selection failed (%s), falling back to profile defaults", e
            )
            return None
        engine = _read(prediction, "engine")
        return engine if _engine_is_available(engine) else None

    def _select_params(self, engine: str, query: str, chunks: list[str]) -> dict | None:
        """Pick ``engine``'s tunable params via DSPy."""
        signature = _PARAM_SIGNATURES.get(engine)
        if self._llm_config is None or signature is None:
            return None
        lm = self.dspy_lm
        try:
            prediction = self._run(
                lm,
                signature,
                _PARAM_SELECT_INSTRUCTION,
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
        return _parse_params(engine, prediction)

    def _expand_labels(
        self, query: str, chunks: list[str], current: list[str]
    ) -> list[str]:
        """Propose extra sensitive-entity labels via DSPy."""
        lm = self.dspy_lm
        try:
            prediction = self._run(
                lm,
                _LabelExpandSignature,
                _LABEL_EXPAND_INSTRUCTION,
                adapter=TolerantJSONAdapter(),
                query=query,
                context="\n\n".join(chunks),
                current_entities=list(current),
            )
        except Exception as e:
            logger.warning(
                "Label expansion failed (%s), keeping the current entity set", e
            )
            return []
        return _clean_label_additions(
            getattr(prediction, "additional_entities", None), current
        )

    def _expand_labels_from_feedback(
        self, state: PrivacyState, policy: PrivacyPolicy, feedback: str
    ) -> list[str]:
        """Propose sensitive labels that act on the reviewer's guidance."""
        if self._llm_config is None or not feedback:
            return []
        current = list(policy.sensitive_entities)
        lm = self.dspy_lm
        try:
            prediction = self._run(
                lm,
                _FeedbackLabelExpandSignature,
                _FEEDBACK_LABEL_EXPAND_INSTRUCTION,
                adapter=TolerantJSONAdapter(),
                query=state.get("query", ""),
                context="\n\n".join(state.get("raw_chunks", [])),
                current_entities=current,
                feedback=feedback,
            )
        except Exception as e:
            logger.warning("Feedback-driven label expansion failed (%s)", e)
            return []
        return _clean_label_additions(
            getattr(prediction, "additional_entities", None), current
        )

    # -- Policy construction ---------------------------------------------

    def _resolve_engine(self, requested: str, profile_default: str) -> str:
        """Validate the engine pinned in the config."""
        if _engine_is_available(requested):
            return requested
        logger.warning(
            "The given engine (%s) cannot be resolved, falling back to default (%s)",
            requested,
            profile_default,
        )
        return profile_default

    def build_policy(self, query: str, chunks: list[str]) -> PrivacyPolicy:
        """Resolve the domain and merge profile defaults with config overrides."""
        domain = self.config.domain or self._infer_domain(query, chunks)
        profile = get_domain_profile(domain)
        detection_cfg = self.config.detection
        sanitization_cfg = self.config.sanitization

        if detection_cfg.engine:
            engine = self._resolve_engine(detection_cfg.engine, profile.default_engine)
        else:
            engine = self._select_engine(query, chunks) or profile.default_engine

        defaults = asdict(DETECTION_DEFAULT_PARAMS[engine])
        if detection_cfg.confidence_threshold is not None:
            detection_params = defaults | {
                "confidence_threshold": detection_cfg.confidence_threshold
            }
        else:
            detection_params = self._select_params(engine, query, chunks) or defaults

        if detection_cfg.entity_types:
            sensitive_entities = list(detection_cfg.entity_types)
        else:
            sensitive_entities = list(profile.sensitive_entities)
            sensitive_entities += self._expand_labels(query, chunks, sensitive_entities)

        return PrivacyPolicy(
            domain=domain,
            sensitive_entities=sensitive_entities,
            detection_engine=engine,
            detection_params=detection_params,
            sanitization_strategy=(
                sanitization_cfg.strategy or profile.default_strategy
            ),
            consistency=(
                profile.default_consistency
                if sanitization_cfg.consistency is None
                else sanitization_cfg.consistency
            ),
            domain_prompt=profile.domain_prompt,
            sanitizer_system_prompt=profile.sanitizer_system_prompt,
        )

    # -- Escalation ------------------------------------------------------

    def _operator_params(self, operator: str) -> dict[str, object]:
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

    def _apply_feedback_overrides(
        self, policy: PrivacyPolicy, feedback: str, respect_config_pins: bool
    ) -> PrivacyPolicy:
        """Switch engine/strategy/threshold per the guidance; pins bind the adversary only."""
        if self._llm_config is None or not feedback:
            return policy
        lm = self.dspy_lm
        try:
            prediction = self._run(
                lm,
                _FeedbackPolicySignature,
                _FEEDBACK_POLICY_INSTRUCTION,
                feedback=feedback,
                engine_guidance=format_guidance(DETECTION_GUIDANCE),
                strategy_guidance=format_guidance(SANITIZATION_GUIDANCE),
                operator_guidance=format_guidance(PRESIDIO_OPERATOR_GUIDANCE),
                available_engines=list(DETECTION_TOOL_NAMES),
                available_strategies=list(SANITIZATION_TOOL_NAMES),
            )
        except Exception as e:
            logger.warning("Feedback policy override failed (%s)", e)
            return policy

        pins = self.config if respect_config_pins else None
        updates: dict[str, object] = {}
        params = dict(policy.detection_params)
        params_changed = False

        engine = _read(prediction, "requested_engine")
        if engine in DETECTION_TOOL_NAMES and not (pins and pins.detection.engine):
            updates["detection_engine"] = engine
            if engine != policy.detection_engine:
                threshold = params.get("confidence_threshold")
                params = asdict(DETECTION_DEFAULT_PARAMS[engine])
                if threshold is not None:
                    params["confidence_threshold"] = threshold
                params_changed = True

        strategy = _read(prediction, "requested_strategy")
        if strategy in SANITIZATION_TOOL_NAMES and not (
            pins and pins.sanitization.strategy
        ):
            updates["sanitization_strategy"] = strategy

        threshold_level = _read(prediction, "requested_threshold")
        if threshold_level in THRESHOLD_LEVELS and not (
            pins and pins.detection.confidence_threshold is not None
        ):
            params["confidence_threshold"] = THRESHOLD_LEVELS[threshold_level]
            params_changed = True

        if params_changed:
            updates["detection_params"] = params

        # Operator and rewrite instruction only apply under their own strategy
        final_strategy = updates.get(
            "sanitization_strategy", policy.sanitization_strategy
        )
        operator = _read(prediction, "requested_operator")
        if final_strategy == "presidio" and operator in PRESIDIO_OPERATOR_GUIDANCE:
            updates["sanitization_params"] = {
                "operator": operator,
                "operator_params": self._operator_params(operator),
            }
        if final_strategy == "synthetic_rewrite":
            prompt = _pin_instruction(
                policy.sanitizer_system_prompt,
                str(getattr(prediction, "rewrite_instruction", "")).strip(),
            )
            if prompt != policy.sanitizer_system_prompt:
                updates["sanitizer_system_prompt"] = prompt

        if updates.get("detection_engine", policy.detection_engine) == "llm":
            prompt = _pin_instruction(
                policy.detector_system_prompt,
                str(getattr(prediction, "detection_instruction", "")).strip(),
            )
            if prompt != policy.detector_system_prompt:
                updates["detector_system_prompt"] = prompt

        return replace(policy, **updates) if updates else policy

    def _apply_guidance(
        self,
        state: PrivacyState,
        policy: PrivacyPolicy,
        guidance: str,
        guidance_label: str,
        respect_config_pins: bool,
    ) -> PrivacyPolicy:
        """Act on a reviewer's guidance text; note-only hardening without one."""
        if not guidance:
            return harden(policy)
        extra_entities = self._expand_labels_from_feedback(state, policy, guidance)
        hardened = harden(policy, extra_entities or None, guidance, guidance_label)
        return self._apply_feedback_overrides(hardened, guidance, respect_config_pins)

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
                state, policy, report, "Human guidance", respect_config_pins=False
            )
            from_human_feedback = bool(report)
        elif verdict is not None and verdict.leaked:
            report = (verdict.recommendation or "").strip()
            new_policy = self._apply_guidance(
                state, policy, report, "Adversary guidance", respect_config_pins=True
            )
            trigger_vector, trigger_entity = verdict.vector, verdict.entity_type
            adversary_escalations += 1
            budget = self.config.leakage_adversary.max_iterations
            vector = (
                trigger_vector.value.replace("_", " ") if trigger_vector else "a probe"
            )
            reason = (
                f"Leak {adversary_escalations}/{budget}: the adversary recovered "
                f"{trigger_entity or 'sensitive data'} via {vector} "
                f"(confidence {verdict.confidence:.2f})"
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
            escalation_log=[*state.get("escalation_log", []), record],
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
            return PrivacyState(
                policy=self.build_policy(
                    state.get("query", ""), list(state.get("raw_chunks", []))
                )
            )
        return self._escalate(state, policy)
