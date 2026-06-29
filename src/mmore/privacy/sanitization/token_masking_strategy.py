"""Token-masking sanitization strategy.

Replaces each detected PII span with an ``[LABEL_N]`` token. When
``policy.consistency`` is true, the same original text always maps to the
same token within a single ``apply`` call.
"""

from typing import Dict, List, Tuple

from ..agents.registry import register_tool
from ..detection.base import PIISpan
from ..policy import PrivacyPolicy
from .base import SanitizationStrategy, apply_replacements, select_non_overlapping


class TokenMaskingStrategy(SanitizationStrategy):
    """Replace each span with ``[LABEL_N]`` (N counts per label)."""

    def apply(
        self,
        chunks: List[str],
        spans_per_chunk: List[List[PIISpan]],
        policy: PrivacyPolicy,
    ) -> List[str]:
        consistency = bool(policy.consistency)
        counters: Dict[str, int] = {}
        memory: Dict[Tuple[str, str], str] = {}

        def token_for(span: PIISpan, original: str) -> str:
            if consistency:
                key = (span.label, original)
                cached = memory.get(key)
                if cached is not None:
                    return cached
            counters[span.label] = counters.get(span.label, 0) + 1
            token = f"[{span.label}_{counters[span.label]}]"
            if consistency:
                memory[(span.label, original)] = token
            return token

        out: List[str] = []
        for chunk, spans in zip(chunks, spans_per_chunk):
            kept = select_non_overlapping(list(spans))
            out.append(apply_replacements(chunk, kept, token_for))
        return out


@register_tool("sanitize_token_masking")
def sanitize_token_masking(
    chunks: List[str],
    spans_per_chunk: List[List[PIISpan]],
    policy: PrivacyPolicy,
) -> List[str]:
    """Apply the default-configured token-masking strategy."""
    return TokenMaskingStrategy().apply(chunks, spans_per_chunk, policy)
