"""LLM-backed PII detection engine using DSPy for typed structured output."""

import logging
from typing import Any, List, Optional, Sequence

import dspy
from pydantic import BaseModel, Field
from typing_extensions import Self

from ...rag.llm import LLMConfig
from ..agents.registry import register_tool
from ..dspy_llm import build_dspy_lm
from .base import DetectionEngine, PIISpan
from .config import DetectionConfig
from .defaults import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_LABELS,
    DEFAULT_LLM_CONFIG,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

PII_DETECTION_INSTRUCTION = (
    "Find every PII occurrence in the input text. For each, return the\n"
    "exact substring (not paraphrased), its entity label, and a confidence."
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
    label: str = Field(description=SPAN_LABEL_DESC)
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=SPAN_SCORE_DESC,
    )


def _build_signature() -> Any:
    class DetectPIISignature(dspy.Signature):
        text: str = dspy.InputField(desc=INPUT_TEXT_DESC)
        entity_types: List[str] = dspy.InputField(desc=INPUT_ENTITY_TYPES_DESC)
        spans: List[_DetectedSpan] = dspy.OutputField(desc=OUTPUT_SPANS_DESC)

    return DetectPIISignature.with_instructions(PII_DETECTION_INSTRUCTION)


# TODO: refine these examples later once we have domain definitions
def _build_demos() -> List[Any]:
    return [
        dspy.Example(
            text="John Doe called from 555-1234 about his MRN 87654321.",
            entity_types=list(DEFAULT_LABELS),
            spans=[
                _DetectedSpan(text="John Doe", label="PERSON", score=0.95),
                _DetectedSpan(text="555-1234", label="PHONE", score=0.95),
                _DetectedSpan(text="87654321", label="MRN", score=0.95),
            ],
        ).with_inputs("text", "entity_types"),
        dspy.Example(
            text="Patient at 123 Main St emailed jane@example.com on 2024-01-15.",
            entity_types=list(DEFAULT_LABELS),
            spans=[
                _DetectedSpan(text="123 Main St", label="LOCATION", score=0.9),
                _DetectedSpan(text="jane@example.com", label="EMAIL", score=0.95),
                _DetectedSpan(text="2024-01-15", label="DATE", score=0.9),
            ],
        ).with_inputs("text", "entity_types"),
    ]


def _build_predictor() -> dspy.Predict:
    predictor = dspy.Predict(_build_signature())
    predictor.demos = _build_demos()
    return predictor


class LLMDetectionEngine(DetectionEngine):
    """Detect PII spans by prompting an LLM with a typed DSPy signature.

    Each instance carries its own ``LLMConfig``, ``entity_types`` and
    ``confidence_threshold``."""

    def __init__(
        self,
        llm_config: LLMConfig,
        entity_types: Optional[Sequence[str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self._llm_config = llm_config
        self._entity_types: List[str] = (
            list(entity_types) if entity_types else list(DEFAULT_LABELS)
        )
        self._confidence_threshold = confidence_threshold
        self._llm: Optional[dspy.BaseLM] = None
        self._predictor: Optional[dspy.Predict] = None

    @classmethod
    def from_config(cls, config: DetectionConfig) -> Self:
        """Build an engine from a ``DetectionConfig``."""
        if config.llm is None:
            raise ValueError("DetectionConfig.llm must be set")
        return cls(
            llm_config=config.llm,
            entity_types=config.entity_types or None,
            confidence_threshold=config.confidence_threshold,
        )

    @property
    def llm(self) -> dspy.BaseLM:
        """Lazy-build and cache the DSPy LM on first access."""
        llm = self._llm
        if llm is None:
            llm = self._llm = build_dspy_lm(self._llm_config)
        return llm

    @property
    def predictor(self) -> dspy.Predict:
        """Lazy-build and cache the DSPy predictor on first access."""
        predictor = self._predictor
        if predictor is None:
            predictor = self._predictor = _build_predictor()
        return predictor

    def detect(self, text: str) -> List[PIISpan]:
        lm = self.llm
        predictor = self.predictor
        try:
            with dspy.context(lm=lm):
                prediction = predictor(text=text, entity_types=self._entity_types)
        except Exception as e:
            logger.warning("LLM detection failed (%s), returning no spans", e)
            return []

        spans: List[PIISpan] = []
        for s in getattr(prediction, "spans", None) or []:
            try:
                fragment = str(s.text)
                label = str(s.label)
                score = float(s.score)
            except (AttributeError, TypeError, ValueError):
                continue
            if not fragment:
                continue
            score = max(0.0, min(1.0, score))
            if score < self._confidence_threshold:
                continue
            start = text.find(fragment)
            if start < 0:
                logger.debug(
                    "LLM emitted fragment %r not found in source text", fragment
                )
                continue
            spans.append(
                PIISpan(
                    start=start, end=start + len(fragment), label=label, score=score
                )
            )
        return spans


@register_tool("detect_pii_llm")
def detect_pii_llm(text: str) -> List[PIISpan]:
    """Detect PII spans in ``text`` using a default-configured LLM engine.

    Agents needing per-config behavior should be wired by setup code that
    builds an ``LLMDetectionEngine.from_config(detection_cfg)`` and registers
    its ``detect()`` function under a distinct tool name, e.g.::

        engine = LLMDetectionEngine.from_config(detection_cfg)
        register_tool("detect_pii_llm_custom", engine.detect)
    """
    return LLMDetectionEngine(DEFAULT_LLM_CONFIG).detect(text)
