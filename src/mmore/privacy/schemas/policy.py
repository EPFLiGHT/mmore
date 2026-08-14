"""The request-specific privacy policy.

Emitted by the Context/Policy Analyzer and consumed downstream by the
agents in the system (Detector, Sanitizer and Adversarial agents).
"""

from dataclasses import dataclass, field, replace

_ESCALATION_NOTE = (
    "Escalation: a prior sanitization pass leaked, or was not in line with the "
    "user's needs."
)


@dataclass
class PrivacyPolicy:
    """Resolved privacy rules for a single retrieval request."""

    domain: str
    sensitive_entities: list[str]
    detection_engine: str
    sanitization_strategy: str
    consistency: bool
    domain_prompt: str
    detection_params: dict = field(default_factory=dict)
    sanitization_params: dict = field(default_factory=dict)
    sanitizer_system_prompt: str = ""
    detector_system_prompt: str = ""
    flagged_fields: list[str] = field(default_factory=list)


def harden(
    policy: PrivacyPolicy,
    extra_entities: list[str] | None = None,
    guidance: str | None = None,
    guidance_label: str = "Human guidance",
) -> PrivacyPolicy:
    """Policy hardening on re-entry: merge entity labels, pin guidance in the prompt."""
    entities = list(policy.sensitive_entities)
    entities += [e for e in extra_entities or [] if e not in entities]

    prompt = policy.domain_prompt
    if _ESCALATION_NOTE not in prompt:
        prompt += " " + _ESCALATION_NOTE
    if guidance:
        note = f"{guidance_label}: {guidance}"
        if note not in prompt:
            prompt += " " + note

    return replace(policy, sensitive_entities=entities, domain_prompt=prompt)
