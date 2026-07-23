"""Sanitization strategy interface.

Each strategy turns a list of chunks plus their detected PII spans into
sanitized chunks. Strategies are registered as agent tools so the Sanitizer
can resolve them by name from YAML.
"""

from abc import ABC, abstractmethod
from typing import Callable

from ..detection.base import PIISpan
from ..schemas.policy import PrivacyPolicy


class SanitizationStrategy(ABC):
    """Abstract base for sanitization strategies."""

    @abstractmethod
    def apply(
        self,
        chunks: list[str],
        spans_per_chunk: list[list[PIISpan]],
        policy: PrivacyPolicy,
    ) -> list[str]:
        """Return the sanitized version of each chunk."""


# A registered sanitization tool: sanitize every chunk under a policy
SanitizationTool = Callable[[list[str], list[list[PIISpan]], PrivacyPolicy], list[str]]


def select_non_overlapping(spans: list[PIISpan]) -> list[PIISpan]:
    """Keep non-overlapping spans, breaking ties by higher score then longer span.

    Required before text replacement: two overlapping spans cannot both be
    applied to the same region, so the sanitizer must choose one.
    """
    ordered = sorted(
        spans, key=lambda s: (s.score, s.end - s.start, -s.start), reverse=True
    )
    chosen: list[PIISpan] = []
    for span in ordered:
        if any(not (span.end <= c.start or span.start >= c.end) for c in chosen):
            continue
        chosen.append(span)
    return chosen


def apply_replacements(
    text: str,
    spans: list[PIISpan],
    replace: Callable[[PIISpan, str], str],
) -> str:
    """Compute replacements left-to-right, then apply them right-to-left."""
    replacements = [
        (span, replace(span, text[span.start : span.end]))
        for span in sorted(spans, key=lambda s: s.start)
    ]
    for span, value in reversed(replacements):
        text = text[: span.start] + value + text[span.end :]
    return text
