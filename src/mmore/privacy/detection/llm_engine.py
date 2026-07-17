"""LLM-backed PII detection engine using DSPy for typed structured output."""

import logging
from typing import List, Optional, Sequence

import dspy
from pydantic import BaseModel, Field
from typing_extensions import Self

from ...rag.llm import LLMConfig
from ..agents.registry import register_tool
from ..config import DetectionConfig
from ..dspy_llm import build_dspy_lm
from ..policy import PrivacyPolicy
from ..ux import report_notice
from .base import DetectionEngine, PIISpan
from .constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_ENTITIES,
    DEFAULT_LLM_CONFIG,
)

logger = logging.getLogger(__name__)


def _warn(message: str) -> None:
    if not report_notice(message):
        logger.warning(message)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

PII_DETECTION_INSTRUCTION = (
    "Find every PII occurrence in the input text. For each, return the "
    "exact substring (not paraphrased), its entity label, and a confidence. "
    "Return spans in the order they appear in the text."
)

SPAN_TEXT_DESC = "exact substring of the input that is PII"
SPAN_LABEL_DESC = "entity type label, e.g. PERSON, EMAIL, MRN"
SPAN_SCORE_DESC = "confidence in [0, 1]; not calibrated, but constrained to the range"

INPUT_TEXT_DESC = "text to scan for PII"
INPUT_ENTITY_TYPES_DESC = "restrict detection to these entity type labels"
OUTPUT_SPANS_DESC = (
    "list of detected PII spans, each with the exact substring from the input"
)


class _DetectedSpan(BaseModel):
    text: str = Field(description=SPAN_TEXT_DESC)
    label: Optional[str] = Field(default=None, description=SPAN_LABEL_DESC)
    score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=SPAN_SCORE_DESC,
    )


class _DetectPIISignature(dspy.Signature):
    text: str = dspy.InputField(desc=INPUT_TEXT_DESC)
    entity_types: List[str] = dspy.InputField(desc=INPUT_ENTITY_TYPES_DESC)
    spans: List[_DetectedSpan] = dspy.OutputField(desc=OUTPUT_SPANS_DESC)


def _build_demos() -> List[dspy.Example]:
    return [
        dspy.Example(
            text="John Doe called from 555-1234 about his MRN 87654321.",
            entity_types=list(DEFAULT_ENTITIES),
            spans=[
                _DetectedSpan(text="John Doe", label="PERSON", score=0.95),
                _DetectedSpan(text="555-1234", label="PHONE_NUMBER", score=0.95),
                _DetectedSpan(text="87654321", label="MRN", score=0.95),
            ],
        ).with_inputs("text", "entity_types"),
        dspy.Example(
            text="Patient at 123 Main St emailed jane@example.com on 2024-01-15.",
            entity_types=list(DEFAULT_ENTITIES),
            spans=[
                _DetectedSpan(text="123 Main St", label="LOCATION", score=0.9),
                _DetectedSpan(
                    text="jane@example.com", label="EMAIL_ADDRESS", score=0.95
                ),
                _DetectedSpan(text="2024-01-15", label="DATE_TIME", score=0.9),
            ],
        ).with_inputs("text", "entity_types"),
    ]


def _build_predictor(instruction: str = "") -> dspy.Predict:
    full = f"{PII_DETECTION_INSTRUCTION} {instruction}".strip()
    predictor = dspy.Predict(_DetectPIISignature.with_instructions(full))
    predictor.demos = _build_demos()
    return predictor


class LLMDetectionEngine(DetectionEngine):
    """Detect PII spans by prompting an LLM with a typed DSPy signature.

    Each instance carries its own ``LLMConfig``, ``sensitive_entities`` and
    ``confidence_threshold``."""

    def __init__(
        self,
        llm_config: LLMConfig,
        sensitive_entities: Optional[Sequence[str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        instruction: str = "",
    ):
        self._llm_config = llm_config
        self._sensitive_entities: List[str] = (
            list(sensitive_entities) if sensitive_entities else list(DEFAULT_ENTITIES)
        )
        self._confidence_threshold = confidence_threshold
        self._instruction = instruction
        self._llm: Optional[dspy.BaseLM] = None
        self._predictor: Optional[dspy.Predict] = None

    @classmethod
    def from_config(cls, config: DetectionConfig) -> Self:
        """Build an engine from a ``DetectionConfig``."""
        llm_config = config.llm
        if llm_config is None:
            llm_config = DEFAULT_LLM_CONFIG
            logger.warning(
                "DetectionConfig.llm not set, falling back to default LLM %r",
                DEFAULT_LLM_CONFIG.llm_name,
            )
        return cls(
            llm_config=llm_config,
            sensitive_entities=config.entity_types or None,
            confidence_threshold=(
                config.confidence_threshold
                if config.confidence_threshold is not None
                else DEFAULT_CONFIDENCE_THRESHOLD
            ),
        )

    @property
    def llm(self) -> dspy.BaseLM:
        """Lazy-build and cache the DSPy LM on first access."""
        if self._llm is None:
            self._llm = build_dspy_lm(self._llm_config)
        return self._llm

    @property
    def predictor(self) -> dspy.Predict:
        """Lazy-build and cache the DSPy predictor on first access."""
        if self._predictor is None:
            self._predictor = _build_predictor(self._instruction)
        return self._predictor

    def detect(self, text: str) -> List[PIISpan]:
        lm = self.llm
        predictor = self.predictor
        try:
            with dspy.context(lm=lm):
                prediction = predictor(text=text, entity_types=self._sensitive_entities)
        except Exception as e:
            logger.debug("LLM detection failed: %s", e)
            _warn(
                "Detector: the LLM answer could not be read, this chunk was left "
                "unscanned (a rule-based engine like presidio is steadier here)"
            )
            return []

        spans: List[PIISpan] = []
        unusable = 0
        # Maps repeated fragments to successive occurrences
        search_cursors: dict[str, int] = {}
        for s in getattr(prediction, "spans", None) or []:
            try:
                fragment = str(s.text)
                label = str(s.label) if s.label else ""
                score = float(s.score)
            except (AttributeError, TypeError, ValueError):
                unusable += 1
                continue
            if not fragment or not label:
                unusable += 1
                continue
            score = max(0.0, min(1.0, score))
            if score < self._confidence_threshold:
                continue
            start = text.find(fragment, search_cursors.get(fragment, 0))
            if start < 0:
                # Paraphrased instead of quoted: nothing to mask in the source
                logger.debug("LLM span %r is not in the source text", fragment)
                unusable += 1
                continue
            search_cursors[fragment] = start + len(fragment)
            spans.append(
                PIISpan(
                    start=start, end=start + len(fragment), label=label, score=score
                )
            )
        if unusable:
            _warn(
                f"Detector: the LLM returned {unusable} unusable span(s), "
                f"ignored them and kept {len(spans)}"
            )
        return spans


@register_tool("detect_pii_llm")
def detect_pii_llm(text: str, policy: PrivacyPolicy) -> List[PIISpan]:
    """Detect PII spans in ``text`` using an LLM engine configured from ``policy``.

    The policy's ``sensitive_entities`` set the engine's entity labels and
    ``detection_params`` (e.g. ``confidence_threshold``) are forwarded to the
    engine constructor. The LLM backend uses ``DEFAULT_LLM_CONFIG`` since
    the policy does not carry an ``LLMConfig``; setup code wanting a custom
    LLM should build ``LLMDetectionEngine.from_config(detection_cfg)`` and
    register its ``detect()`` under a distinct tool name.
    """
    engine = LLMDetectionEngine(
        DEFAULT_LLM_CONFIG,
        sensitive_entities=policy.sensitive_entities or None,
        instruction=policy.detector_system_prompt,
        **policy.detection_params,
    )
    return engine.detect(text)
