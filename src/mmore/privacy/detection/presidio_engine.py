"""Presidio-based PII detection engine.

Wraps ``presidio_analyzer.AnalyzerEngine`` (rule-based and spaCy NER) with
possibility to add custom clinical recognizers.
"""

import importlib
import logging
import threading
from typing import Any, List, Optional, Sequence

from typing_extensions import Self

from ..agents.registry import register_tool
from .base import DetectionEngine, PIISpan
from .config import DetectionConfig
from .defaults import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_PRESIDIO_SPACY_MODEL,
    PRESIDIO_CLINICAL_PATTERNS,
)

logger = logging.getLogger(__name__)


def _ensure_spacy_model(model_name: str) -> None:
    """Make sure the spaCy model for Presidio's NLP engine is installed."""
    import spacy

    if spacy.util.is_package(model_name):
        return
    logger.warning(
        "spaCy model %r not found, downloading it...",
        model_name,
    )
    spacy.cli.download(model_name)
    importlib.invalidate_caches()


def _build_clinical_recognizers() -> List[Any]:
    """Build the clinical-domain custom recognizers."""
    from presidio_analyzer import Pattern, PatternRecognizer

    recognizers: List[Any] = []
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


def _load_presidio_analyzer() -> Any:
    """Build a ``presidio_analyzer.AnalyzerEngine`` with custom clinical recognizers."""
    from presidio_analyzer import AnalyzerEngine

    _ensure_spacy_model(DEFAULT_PRESIDIO_SPACY_MODEL)
    analyzer = AnalyzerEngine()
    for recognizer in _build_clinical_recognizers():
        analyzer.registry.add_recognizer(recognizer)
    return analyzer


_analyzer_cache: Optional[Any] = None
_analyzer_cache_lock = threading.Lock()


def _get_or_load_analyzer() -> Any:
    global _analyzer_cache
    if _analyzer_cache is not None:
        return _analyzer_cache
    with _analyzer_cache_lock:
        if _analyzer_cache is None:
            _analyzer_cache = _load_presidio_analyzer()
        return _analyzer_cache


def clear_presidio_cache() -> None:
    """Drop the cached analyzer."""
    global _analyzer_cache
    with _analyzer_cache_lock:
        _analyzer_cache = None


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
    def analyzer(self) -> Any:
        return _get_or_load_analyzer()

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
