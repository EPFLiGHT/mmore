"""Shared constants for the sanitization strategies."""

# Strategy names used in YAML configs mapped to the tool names
SANITIZATION_TOOL_NAMES: dict[str, str] = {
    "token_masking": "sanitize_token_masking",
    "entity_replacement": "sanitize_entity_replacement",
    "synthetic_rewrite": "sanitize_synthetic_rewrite",
    "presidio": "sanitize_presidio",
}


# Per-strategy guidance so the analyzer can map descriptive feedback to a choice
SANITIZATION_GUIDANCE: dict[str, str] = {
    "token_masking": (
        "replace each detected entity with a typed placeholder like [PERSON_1], "
        "the original value is removed entirely"
    ),
    "entity_replacement": (
        "swap each entity for a realistic fake value (Faker), consistent per "
        "subject across chunks, keeps the text natural while hiding the real value"
    ),
    "synthetic_rewrite": (
        "an LLM rewrites the passage so it carries no identifiers while "
        "preserving the domain meaning; the most flexible strategy, it can "
        "follow custom instructions on how the sanitized text should read"
    ),
    "presidio": "apply Presidio's built-in anonymization operators to detected spans",
}


# Per-operator guidance so the analyzer can map descriptive feedback to a choice
PRESIDIO_OPERATOR_GUIDANCE: dict[str, str] = {
    "replace": "swap each value for its entity label, e.g. <PERSON>",
    "redact": "delete each value entirely, leaving no placeholder",
    "mask": "overwrite each value with a masking character like ***",
    "hash": "replace each value with its hash, irreversible but consistent",
    "encrypt": "AES-encrypt each value, recoverable with the key",
}
