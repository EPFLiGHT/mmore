"""Context/Policy Analyzer.

Pipeline:  [analyzer]  ->  detector  ->  sanitizer  ->  ... TODO: complete once implemented
Reads:     query, raw_chunks
Writes:    policy

One node in the privacy multi-agent pipeline: picks the privacy domain
(explicit config or inferred from the query and raw chunks) and emits
the per-request PrivacyPolicy the next agents consume.
"""

import logging
import re
from dataclasses import asdict
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
from ..dspy_llm import build_dspy_lm
from ..policy import PrivacyPolicy
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


# ========================================================================
# DSPy signature field descriptions
# ========================================================================

_QUERY_DESC = "the user request"
_CONTEXT_DESC = "the retrieved context (concatenation of the raw chunks)"
_DOMAIN_DESC = "exactly one of: global, healthcare, humanitarian"
_DETECTION_GUIDANCE_DESC = "per-engine guidance: pros, cons, and when to prefer each"
_ENGINE_OUTPUT_DESC = "exactly one of: presidio, gliner, openai, llm"
_PARAM_GUIDANCE_DESC = "engine-specific guidance for each tunable parameter"
_THRESHOLD_OUTPUT_DESC = "exactly one of: low, medium, high"
_CURRENT_ENTITIES_DESC = "the sensitive entity labels already in the policy"
_ADDITIONAL_ENTITIES_DESC = (
    "JSON array of uppercase identifier strings, e.g. "
    '["PASSPORT_NUMBER", "BANK_ACCOUNT"]. Empty array if nothing extra is needed.'
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
    engine: Literal["presidio", "gliner", "openai", "llm"] = dspy.OutputField(
        desc=_ENGINE_OUTPUT_DESC
    )


class _LabelExpandSignature(dspy.Signature):
    query: str = dspy.InputField(desc=_QUERY_DESC)
    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    current_entities: List[str] = dspy.InputField(desc=_CURRENT_ENTITIES_DESC)
    additional_entities: List[str] = dspy.OutputField(desc=_ADDITIONAL_ENTITIES_DESC)


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
    "openai": _OpenAIFilterParamsSignature,
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


def _format_engine_guidance() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in DETECTION_GUIDANCE.items())


def _parse_param_prediction(  # TODO: check once final implemented new parameters to add
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
        config: Union[PrivacyConfig, str, dict],
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
            with dspy.context(lm=self._dspy_lm, adapter=dspy.JSONAdapter()):
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

    def _node(self, state: PrivacyState) -> PrivacyState:
        """Graph node: write the resolved policy into the pipeline state."""
        policy = self.build_policy(
            state.get("query", ""), list(state.get("raw_chunks", []))
        )
        return PrivacyState(policy=policy)
