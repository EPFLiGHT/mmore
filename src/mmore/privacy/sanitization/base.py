"""Sanitization strategy interface.

Each strategy turns a list of chunks plus their detected PII spans into
sanitized chunks. Strategies are registered as agent tools so the Sanitizer
can resolve them by name from YAML.
"""

from abc import ABC, abstractmethod
from typing import Callable, List

from ..detection.base import PIISpan
from ..policy import PrivacyPolicy


class SanitizationStrategy(ABC):
    """Abstract base for sanitization strategies."""

    @abstractmethod
    def apply(
        self,
        chunks: List[str],
        spans_per_chunk: List[List[PIISpan]],
        policy: PrivacyPolicy,
    ) -> List[str]:
        """Return the sanitized version of each chunk."""


def select_non_overlapping(spans: List[PIISpan]) -> List[PIISpan]:
    """Keep non-overlapping spans, breaking ties by higher score then longer span.

    Required before text replacement: two overlapping spans cannot both be
    applied to the same region, so the sanitizer must choose one.
    """
    ordered = sorted(
        spans,
        key=lambda s: (s.score, s.end - s.start, -s.start),
        reverse=True,
    )

    chosen: List[PIISpan] = []

    for span in ordered:
        # check overlap with already selected spans
        if any(not (span.end <= c.start or span.start >= c.end) for c in chosen):
            continue

        chosen.append(span)

    return chosen


def apply_replacements(
    text: str,
    spans: List[PIISpan],
    replace: Callable[[PIISpan, str], str],
) -> str:
    """Compute replacements left-to-right, then apply them in right-to-left."""
    ordered = sorted(spans, key=lambda s: s.start)
    replacements = [
        (span, replace(span, text[span.start : span.end])) for span in ordered
    ]

    for span, value in reversed(replacements):
        text = text[: span.start] + value + text[span.end :]

    return text
