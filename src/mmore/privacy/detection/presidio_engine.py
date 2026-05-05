"""Presidio-based PII detection engine.

Wraps ``presidio_analyzer.AnalyzerEngine`` (rule-based and spaCy NER) with
possibility to add custom clinical recognizers.
"""

import threading
from typing import Any, List, Optional, Sequence

from typing_extensions import Self

from ..agents.registry import register_tool
from .base import DetectionEngine, PIISpan
from .config import DetectionConfig


def _build_clinical_recognizers() -> List[Any]:
    """Build the clinical-domain custom recognizers (MRN, hospital dates, insurance ID)."""
    from presidio_analyzer import Pattern, PatternRecognizer

    mrn = PatternRecognizer(
        supported_entity="MRN",
        patterns=[
            Pattern(
                name="mrn_with_prefix",
                regex=r"\bMRN[\s:#]*\d{6,10}\b",
                score=0.9,
            ),
            Pattern(
                name="mrn_bare_8_digits",
                regex=r"\b\d{8}\b",
                score=0.4,
            ),
        ],
        context=["mrn", "medical record", "record number", "patient id"],
    )

    hospital_date = PatternRecognizer(
        supported_entity="HOSPITAL_DATE",
        patterns=[
            Pattern(
                name="iso_date",
                regex=r"\b\d{4}-\d{2}-\d{2}\b",
                score=0.6,
            ),
            Pattern(
                name="us_date",
                regex=r"\b\d{1,2}/\d{1,2}/\d{4}\b",
                score=0.6,
            ),
        ],
        context=["admission", "discharge", "appointment", "hospital", "clinic"],
    )

    insurance_id = PatternRecognizer(
        supported_entity="INSURANCE_ID",
        patterns=[
            Pattern(
                name="insurance_alnum",
                regex=r"\b[A-Z]{2,3}\d{6,12}\b",
                score=0.7,
            ),
        ],
        context=["insurance", "policy", "member id", "subscriber"],
    )

    # TODO: check how relevant these patterns are and add more
    # TODO: check if we could register a tool to add new patterns
    return [mrn, hospital_date, insurance_id]


def _load_presidio_analyzer() -> Any:
    """Build a ``presidio_analyzer.AnalyzerEngine`` with custom clinical recognizers."""
    from presidio_analyzer import AnalyzerEngine

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
        confidence_threshold: float = 0.7,
        language: str = "en",
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
