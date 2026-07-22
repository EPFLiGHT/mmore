"""LLM-driven synthetic-rewrite sanitization strategy.

Rewrites each chunk via a typed DSPy predictor. The LM is taken from the
current ``dspy.context``.
"""

import logging

import dspy

from ..agents.registry import register_tool
from ..detection.base import PIISpan
from ..schemas.policy import PrivacyPolicy
from .base import SanitizationStrategy

logger = logging.getLogger(__name__)


_REWRITE_INSTRUCTION = (
    "Rewrite the chunk so it carries no sensitive personal identifiers while "
    "preserving the factual and topical content needed downstream. Follow the "
    "domain-specific sanitization guidance in the system prompt when one is "
    "given. The detected_entities list flags PII already found in the chunk: "
    "remove or generalize each one."
)


class _RewriteSignature(dspy.Signature):
    system_prompt: str = dspy.InputField(
        desc="domain-specific sanitization guidance for the rewrite, may be empty"
    )
    detected_entities: str = dspy.InputField(
        desc="newline-separated 'LABEL: text' of PII already detected in the chunk"
    )
    chunk: str = dspy.InputField(desc="the raw chunk to sanitize")
    sanitized: str = dspy.OutputField(
        desc="the sanitized rewrite of the chunk, preserving factual content"
    )


def _format_entities(chunk: str, spans: list[PIISpan]) -> str:
    """Render spans as newline-separated ``LABEL: text`` for the predictor."""
    return "\n".join(f"{span.label}: {chunk[span.start : span.end]}" for span in spans)


class SyntheticRewriteStrategy(SanitizationStrategy):
    """Rewrite each chunk via DSPy."""

    def apply(
        self,
        chunks: list[str],
        spans_per_chunk: list[list[PIISpan]],
        policy: PrivacyPolicy,
    ) -> list[str]:
        predictor = dspy.Predict(
            _RewriteSignature.with_instructions(_REWRITE_INSTRUCTION)
        )
        sanitized: list[str] = []
        for number, (chunk, spans) in enumerate(zip(chunks, spans_per_chunk), 1):
            if not spans:
                sanitized.append(chunk)
                continue
            logger.debug("Synthetic rewrite (LLM): chunk %d/%d", number, len(chunks))
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
                sanitized.append(chunk)
                continue
            rewritten = str(getattr(prediction, "sanitized", "")).strip()
            sanitized.append(rewritten or chunk)
        return sanitized


@register_tool("sanitize_synthetic_rewrite")
def sanitize_synthetic_rewrite(
    chunks: list[str],
    spans_per_chunk: list[list[PIISpan]],
    policy: PrivacyPolicy,
) -> list[str]:
    """Apply the default synthetic-rewrite strategy; needs an LM in ``dspy.context``."""
    return SyntheticRewriteStrategy().apply(chunks, spans_per_chunk, policy)
