"""Presidio-based sanitization strategy.

Delegates sanitization to ``presidio_anonymizer.AnonymizerEngine``. Detected
PII spans are converted to ``RecognizerResult`` records and replaced with
``<LABEL>`` placeholders by default.
"""

import logging
from enum import Enum
from typing import TYPE_CHECKING, List, Union

from .._cache import MODEL_REGISTRY
from ..agents.registry import register_tool
from ..detection.base import PIISpan
from ..policy import PrivacyPolicy
from .base import SanitizationStrategy, select_non_overlapping

if TYPE_CHECKING:
    from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "presidio_anonymizer"


class PresidioOperator(str, Enum):
    """Supported Presidio ``AnonymizerEngine`` operators.

    See https://microsoft.github.io/presidio/anonymizer/ for more info.
    """

    REPLACE = "replace"
    REDACT = "redact"
    MASK = "mask"
    HASH = "hash"
    ENCRYPT = "encrypt"
    KEEP = "keep"
    CUSTOM = "custom"


DEFAULT_OPERATOR = PresidioOperator.REPLACE


def _load_anonymizer() -> "AnonymizerEngine":
    from presidio_anonymizer import AnonymizerEngine

    return AnonymizerEngine()


def _get_or_load_anonymizer() -> "AnonymizerEngine":
    """Lazily build and cache a Presidio ``AnonymizerEngine``."""
    return MODEL_REGISTRY.get_or_load(_CACHE_PREFIX, _load_anonymizer)


def clear_presidio_anonymizer_cache() -> None:
    """Drop the cached anonymizer."""
    MODEL_REGISTRY.clear(prefix=_CACHE_PREFIX)


def _normalize_operator(raw: Union[str, PresidioOperator]) -> str:
    """Normalize an operator value (str or ``PresidioOperator``) to its string.

    ``PresidioOperator`` is a ``str`` enum, so its constructor accepts both a
    raw string and one of its own members; anything unknown raises ``ValueError``.
    """
    try:
        return PresidioOperator(raw).value
    except ValueError as error:
        supported = ", ".join(operator.value for operator in PresidioOperator)
        raise ValueError(
            f"Unsupported Presidio operator '{raw}'. Supported: {supported}"
        ) from error


class PresidioSanitizationStrategy(SanitizationStrategy):
    """Sanitize each chunk via Presidio's ``AnonymizerEngine``."""

    def apply(
        self,
        chunks: List[str],
        spans_per_chunk: List[List[PIISpan]],
        policy: PrivacyPolicy,
    ) -> List[str]:
        from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

        anonymizer = _get_or_load_anonymizer()
        params = policy.sanitization_params or {}
        operator = _normalize_operator(params.get("operator", DEFAULT_OPERATOR))
        operator_params = params.get("operator_params", {}) or {}
        operators = {"DEFAULT": OperatorConfig(operator, operator_params)}

        out: List[str] = []
        for chunk, spans in zip(chunks, spans_per_chunk):
            kept = select_non_overlapping(list(spans))
            if not kept:
                out.append(chunk)
                continue
            recognizer_results = [
                RecognizerResult(
                    entity_type=s.label,
                    start=s.start,
                    end=s.end,
                    score=s.score,
                )
                for s in kept
            ]
            try:
                result = anonymizer.anonymize(
                    text=chunk,
                    analyzer_results=recognizer_results,
                    operators=operators,
                )
                out.append(result.text)
            except Exception as e:
                logger.warning(
                    "Presidio anonymize failed (%s), leaving chunk unchanged", e
                )
                out.append(chunk)
        return out


@register_tool("sanitize_presidio")
def sanitize_presidio(
    chunks: List[str],
    spans_per_chunk: List[List[PIISpan]],
    policy: PrivacyPolicy,
) -> List[str]:
    """Apply the default-configured Presidio anonymizer sanitization strategy."""
    return PresidioSanitizationStrategy().apply(chunks, spans_per_chunk, policy)
