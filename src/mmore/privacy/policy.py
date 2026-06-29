"""The request-specific privacy policy.

Emitted by the Context/Policy Analyzer and consumed downstream by the
agents in the system (Detector, Sanitizer and Adversarial agents).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PrivacyPolicy:
    """Resolved privacy rules for a single retrieval request."""

    domain: str
    sensitive_entities: List[str]
    detection_engine: str
    sanitization_strategy: str
    consistency: bool
    domain_prompt: str
    detection_params: dict = field(default_factory=dict)
    sanitization_params: dict = field(default_factory=dict)
    sanitizer_system_prompt: str = ""  # TODO: check if still useful later or delete
    flagged_fields: List[str] = field(default_factory=list)
