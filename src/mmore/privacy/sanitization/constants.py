"""Shared constants for the sanitization strategies."""

from typing import Dict

# Strategy names used in YAML configs mapped to the tool names
SANITIZATION_TOOL_NAMES: Dict[str, str] = {
    "token_masking": "sanitize_token_masking",
    "entity_replacement": "sanitize_entity_replacement",
    "synthetic_rewrite": "sanitize_synthetic_rewrite",
    "presidio": "sanitize_presidio",
}
