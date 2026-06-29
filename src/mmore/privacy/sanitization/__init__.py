"""Sanitization strategies exposed as agent tools."""

from .base import SanitizationStrategy, apply_replacements, select_non_overlapping
from .constants import SANITIZATION_TOOL_NAMES
from .entity_replacement_strategy import (
    EntityReplacementStrategy,
    sanitize_entity_replacement,
)
from .presidio_strategy import (
    PresidioOperator,
    PresidioSanitizationStrategy,
    sanitize_presidio,
)
from .synthetic_rewrite_strategy import (
    SyntheticRewriteStrategy,
    sanitize_synthetic_rewrite,
)
from .token_masking_strategy import TokenMaskingStrategy, sanitize_token_masking

__all__ = [
    "EntityReplacementStrategy",
    "PresidioOperator",
    "PresidioSanitizationStrategy",
    "SANITIZATION_TOOL_NAMES",
    "SanitizationStrategy",
    "SyntheticRewriteStrategy",
    "TokenMaskingStrategy",
    "apply_replacements",
    "sanitize_entity_replacement",
    "sanitize_presidio",
    "sanitize_synthetic_rewrite",
    "sanitize_token_masking",
    "select_non_overlapping",
]
