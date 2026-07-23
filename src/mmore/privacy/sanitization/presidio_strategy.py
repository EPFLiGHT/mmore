"""Presidio-based sanitization strategy.

Delegates sanitization to ``presidio_anonymizer.AnonymizerEngine``. Detected
PII spans are converted to ``RecognizerResult`` records and replaced with
``<LABEL>`` placeholders by default.
"""

import logging
from enum import Enum
from typing import TYPE_CHECKING

from ..agents.registry import register_tool
from ..detection.base import PIISpan
from ..model_cache import MODEL_REGISTRY
from ..schemas.policy import PrivacyPolicy
from .base import SanitizationStrategy, select_non_overlapping

if TYPE_CHECKING:
    from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)

_CACHE_KEY = "presidio_anonymizer"


class PresidioOperator(str, Enum):
    """Supported Presidio ``AnonymizerEngine`` operators.

    See https://microsoft.github.io/presidio/anonymizer/ for more info.
    """

    REPLACE = "replace"  # default
    REDACT = "redact"
    MASK = "mask"
    HASH = "hash"
    ENCRYPT = "encrypt"


DEFAULT_OPERATOR = PresidioOperator.REPLACE


def _load_anonymizer() -> "AnonymizerEngine":
    from presidio_anonymizer import AnonymizerEngine

    return AnonymizerEngine()


def _normalize_operator(raw: str | PresidioOperator) -> str:
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
        chunks: list[str],
        spans_per_chunk: list[list[PIISpan]],
        policy: PrivacyPolicy,
    ) -> list[str]:
        from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

        anonymizer = MODEL_REGISTRY.get_or_load(_CACHE_KEY, _load_anonymizer)
        params = policy.sanitization_params or {}
        operators = {
            "DEFAULT": OperatorConfig(
                _normalize_operator(params.get("operator", DEFAULT_OPERATOR)),
                params.get("operator_params") or {},
            )
        }

        sanitized: list[str] = []
        for chunk, spans in zip(chunks, spans_per_chunk):
            kept = select_non_overlapping(spans)
            if not kept:
                sanitized.append(chunk)
                continue
            try:
                result = anonymizer.anonymize(
                    text=chunk,
                    analyzer_results=[
                        RecognizerResult(
                            entity_type=s.label,
                            start=s.start,
                            end=s.end,
                            score=s.score,
                        )
                        for s in kept
                    ],
                    operators=operators,
                )
            except Exception as e:
                logger.error("Presidio anonymize failed with error %s", e)
                raise RuntimeError("Presidio sanitization failed.") from e
            sanitized.append(result.text)
        return sanitized


@register_tool("sanitize_presidio")
def sanitize_presidio(
    chunks: list[str],
    spans_per_chunk: list[list[PIISpan]],
    policy: PrivacyPolicy,
) -> list[str]:
    """Apply the default-configured Presidio anonymizer sanitization strategy."""
    return PresidioSanitizationStrategy().apply(chunks, spans_per_chunk, policy)
