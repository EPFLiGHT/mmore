"""Token-masking sanitization strategy.

Replaces each detected PII span with an ``[LABEL_N]`` token. When
``policy.consistency`` is true, the same original text always maps to the
same token within a single ``apply`` call.
"""

from collections import Counter

from ..agents.registry import register_tool
from ..detection.base import PIISpan
from ..schemas.policy import PrivacyPolicy
from .base import SanitizationStrategy, apply_replacements, select_non_overlapping


class TokenMaskingStrategy(SanitizationStrategy):
    """Replace each span with ``[LABEL_N]`` (N counts per label)."""

    def apply(
        self,
        chunks: list[str],
        spans_per_chunk: list[list[PIISpan]],
        policy: PrivacyPolicy,
    ) -> list[str]:
        consistent = bool(policy.consistency)
        seen_per_label: Counter[str] = Counter()
        tokens: dict[tuple[str, str], str] = {}

        def token_for(span: PIISpan, original: str) -> str:
            key = (span.label, original)
            if consistent and key in tokens:
                return tokens[key]
            seen_per_label[span.label] += 1
            token = f"[{span.label}_{seen_per_label[span.label]}]"
            if consistent:
                tokens[key] = token
            return token

        return [
            apply_replacements(chunk, select_non_overlapping(spans), token_for)
            for chunk, spans in zip(chunks, spans_per_chunk)
        ]


@register_tool("sanitize_token_masking")
def sanitize_token_masking(
    chunks: list[str],
    spans_per_chunk: list[list[PIISpan]],
    policy: PrivacyPolicy,
) -> list[str]:
    """Apply the default-configured token-masking strategy."""
    return TokenMaskingStrategy().apply(chunks, spans_per_chunk, policy)
