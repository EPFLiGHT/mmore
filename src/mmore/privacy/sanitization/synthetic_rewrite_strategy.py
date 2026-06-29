"""LLM-driven synthetic-rewrite sanitization strategy.

Rewrites each chunk via a typed DSPy predictor. The LM is taken from the
current ``dspy.context`.
"""

import logging
from typing import List

import dspy

from ..agents.registry import register_tool
from ..detection.base import PIISpan
from ..policy import PrivacyPolicy
from .base import SanitizationStrategy

logger = logging.getLogger(__name__)


# ========================================================================
# Prompts
# ========================================================================

_REWRITE_INSTRUCTION = (
    "Rewrite the chunk so it carries no sensitive personal identifiers while "
    "preserving the factual and topical content needed downstream. Follow the "
    "domain-specific sanitization guidance in the system prompt. The "
    "detected_entities list flags PII already found in the chunk: remove or "
    "generalize each one."
)


# ========================================================================
# DSPy signature
# ========================================================================


class _RewriteSignature(dspy.Signature):
    system_prompt: str = dspy.InputField(
        desc="domain-specific sanitization guidance for the rewrite"
    )
    detected_entities: str = dspy.InputField(
        desc="newline-separated 'LABEL: text' of PII already detected in the chunk"
    )
    chunk: str = dspy.InputField(desc="the raw chunk to sanitize")
    sanitized: str = dspy.OutputField(
        desc="the sanitized rewrite of the chunk, preserving factual content"
    )


# ========================================================================
# Predictors and helpers
# ========================================================================


def _build_rewrite_predictor() -> dspy.Predict:
    return dspy.Predict(_RewriteSignature.with_instructions(_REWRITE_INSTRUCTION))


def _format_entities(chunk: str, spans: List[PIISpan]) -> str:
    """Render spans as newline-separated ``LABEL: text`` for the predictor."""
    return "\n".join(f"{span.label}: {chunk[span.start : span.end]}" for span in spans)


class SyntheticRewriteStrategy(SanitizationStrategy):
    """Rewrite each chunk via DSPy."""

    def apply(
        self,
        chunks: List[str],
        spans_per_chunk: List[List[PIISpan]],
        policy: PrivacyPolicy,
    ) -> List[str]:
        predictor = _build_rewrite_predictor()
        out: List[str] = []
        for chunk, spans in zip(chunks, spans_per_chunk):
            if not spans:
                # No PII detected hence nothing to rewrite
                out.append(chunk)
                continue
            try:
                prediction = predictor(
                    system_prompt=policy.sanitizer_system_prompt,
                    detected_entities=_format_entities(chunk, spans),
                    chunk=chunk,
                )
            except Exception as e:
                logger.warning(
                    "Synthetic rewrite failed (%s); leaving chunk unchanged", e
                )
                out.append(chunk)
                continue
            sanitized = str(getattr(prediction, "sanitized", "")).strip()
            out.append(sanitized if sanitized else chunk)
        return out


@register_tool("sanitize_synthetic_rewrite")
def sanitize_synthetic_rewrite(
    chunks: List[str],
    spans_per_chunk: List[List[PIISpan]],
    policy: PrivacyPolicy,
) -> List[str]:
    """Apply the default synthetic-rewrite strategy; needs an LM in ``dspy.context``."""
    return SyntheticRewriteStrategy().apply(chunks, spans_per_chunk, policy)
