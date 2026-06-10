"""Presidio-based PII detection engine.

Wraps ``presidio_analyzer.AnalyzerEngine`` (rule-based and spaCy NER) with
possibility to add custom clinical recognizers.
"""

import importlib
import logging
from typing import TYPE_CHECKING, List, Optional, Sequence

from typing_extensions import Self

from .._cache import MODEL_REGISTRY
from ..agents.registry import register_tool
from .base import DetectionEngine, PIISpan
from .config import DetectionConfig
from .defaults import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_PRESIDIO_SPACY_MODEL,
    PRESIDIO_CLINICAL_PATTERNS,
)

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "presidio"


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

    Each instance carries its own ``entity_types`` and ``confidence_threshold``,
    the analyzer is shared across instances via ``_analyzer_cache``.
    """

    def __init__(
        self,
        entity_types: Optional[Sequence[str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        language: str = DEFAULT_LANGUAGE,
    ):
        self._entity_types: Optional[List[str]] = (
            list(entity_types) if entity_types else None
        )
        self._confidence_threshold = confidence_threshold
        self._language = language

    @classmethod
    def from_config(cls, config: DetectionConfig) -> Self:
        return cls(
            entity_types=config.entity_types or None,
            confidence_threshold=config.confidence_threshold,
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
            entities=self._entity_types,
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
def detect_pii_presidio(text: str) -> List[PIISpan]:
    """Detect PII spans in ``text`` using a default-configured Presidio engine.

    Agents needing per-config behavior should be wired by setup code that
    builds a ``PresidioEngine.from_config(detection_cfg)`` and registers its
    ``detect()`` function under a distinct tool name, e.g.::

        engine = PresidioEngine.from_config(detection_cfg)
        register_tool("detect_pii_presidio_strict", engine.detect)
    """
    return PresidioEngine().detect(text)
