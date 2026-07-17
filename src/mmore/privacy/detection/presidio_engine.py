"""Presidio-based PII detection engine.

Wraps ``presidio_analyzer.AnalyzerEngine`` (rule-based and spaCy NER) with
possibility to add custom clinical recognizers.
"""

import importlib
import logging
from typing import TYPE_CHECKING, List, Optional, Sequence

from typing_extensions import Self

from ...ux import loading_model
from .._cache import MODEL_REGISTRY
from ..agents.registry import register_tool
from ..config import DetectionConfig, DetectionEngineType
from ..domains.profile import PRESIDIO_CLINICAL_PATTERNS
from ..policy import PrivacyPolicy
from .base import DetectionEngine, PIISpan
from .constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_PRESIDIO_SPACY_MODEL,
)

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer

logger = logging.getLogger(__name__)

_CACHE_PREFIX = DetectionEngineType.PRESIDIO.value


def _ensure_spacy_model(model_name: str) -> None:
    """Make sure the spaCy model for Presidio's NLP engine is installed."""
    import spacy

    if spacy.util.is_package(model_name):
        return
    logger.warning(
        "spaCy model %r not found, downloading it...",
        model_name,
    )
    from spacy.cli.download import download

    download(model_name)
    importlib.invalidate_caches()


def _build_clinical_recognizers() -> "List[PatternRecognizer]":
    """Build the clinical-domain custom recognizers."""
    from presidio_analyzer import Pattern, PatternRecognizer

    recognizers: List[PatternRecognizer] = []
    for spec in PRESIDIO_CLINICAL_PATTERNS:
        recognizers.append(
            PatternRecognizer(
                supported_entity=spec["entity"],
                patterns=[
                    Pattern(name=name, regex=regex, score=score)
                    for name, regex, score in spec["patterns"]
                ],
                context=list(spec["context"]),
            )
        )
    return recognizers


def _load_presidio_analyzer() -> "AnalyzerEngine":
    """Build a ``presidio_analyzer.AnalyzerEngine`` with custom clinical recognizers."""
    from presidio_analyzer import AnalyzerEngine

    with loading_model(
        f"the PII detection engine (presidio, {DEFAULT_PRESIDIO_SPACY_MODEL})"
    ):
        _ensure_spacy_model(DEFAULT_PRESIDIO_SPACY_MODEL)
        analyzer = AnalyzerEngine()
        for recognizer in _build_clinical_recognizers():
            analyzer.registry.add_recognizer(recognizer)
    return analyzer


def clear_presidio_cache() -> None:
    """Drop the cached analyzer."""
    MODEL_REGISTRY.clear(prefix=_CACHE_PREFIX)


class PresidioEngine(DetectionEngine):
    """Detect PII spans with Microsoft Presidio + custom clinical recognizers.

    Each instance carries its own ``sensitive_entities`` and
    ``confidence_threshold``. The model is shared across instances via the
    global ``MODEL_REGISTRY``.
    """

    def __init__(
        self,
        sensitive_entities: Optional[Sequence[str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        language: str = DEFAULT_LANGUAGE,
    ):
        self._sensitive_entities: Optional[List[str]] = (
            list(sensitive_entities) if sensitive_entities else None
        )
        self._confidence_threshold = confidence_threshold
        self._language = language

    @classmethod
    def from_config(cls, config: DetectionConfig) -> Self:
        return cls(
            sensitive_entities=config.entity_types or None,
            confidence_threshold=(
                config.confidence_threshold
                if config.confidence_threshold is not None
                else DEFAULT_CONFIDENCE_THRESHOLD
            ),
        )

    @property
    def analyzer(self) -> "AnalyzerEngine":
        return MODEL_REGISTRY.get_or_load(
            f"{_CACHE_PREFIX}:{DEFAULT_PRESIDIO_SPACY_MODEL}", _load_presidio_analyzer
        )

    def detect(self, text: str) -> List[PIISpan]:
        results = self.analyzer.analyze(
            text=text,
            language=self._language,
            entities=self._sensitive_entities,
            score_threshold=self._confidence_threshold,
        )
        return [
            PIISpan(
                start=int(r.start),
                end=int(r.end),
                label=str(r.entity_type),
                score=float(r.score),
            )
            for r in results
        ]


@register_tool("detect_pii_presidio")
def detect_pii_presidio(text: str, policy: PrivacyPolicy) -> List[PIISpan]:
    """Detect PII spans in ``text`` using a Presidio engine configured from ``policy``."""
    engine = PresidioEngine(
        sensitive_entities=policy.sensitive_entities or None,
        **policy.detection_params,
    )
    return engine.detect(text)
