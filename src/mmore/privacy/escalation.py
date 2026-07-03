"""Policy hardening on re-entry: merge entity labels, pin guidance in the prompt."""

from dataclasses import replace
from typing import List, Optional

from .policy import PrivacyPolicy

_ESCALATION_NOTE = (
    "Escalation: a prior sanitization pass leaked: treat every quasi-identifier "
    "and rare attribute as sensitive and redact aggressively."
)


def apply_entity_guidance(
    policy: PrivacyPolicy,
    extra_entities: Optional[List[str]] = None,
    context_note: Optional[str] = None,
    note_label: str = "Human guidance",
) -> PrivacyPolicy:
    """Merge entity labels; pin the escalation note and guidance into the prompt."""
    entities = list(policy.sensitive_entities) + [
        e for e in (extra_entities or []) if e not in policy.sensitive_entities
    ]
    prompt = policy.domain_prompt
    if _ESCALATION_NOTE not in prompt:
        prompt += " " + _ESCALATION_NOTE
    if context_note:
        guidance = f"{note_label}: {context_note}"
        if guidance not in prompt:
            prompt += " " + guidance
    return replace(policy, sensitive_entities=entities, domain_prompt=prompt)
