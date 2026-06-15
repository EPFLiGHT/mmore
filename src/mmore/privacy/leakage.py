"""The verdict produced by the pre-cloud Leakage Adversary.

Emitted by the ``AdversarialAgent`` after probing the sanitized context
for residual PII and quasi-identifiers, and consumed by the escalation loop
and the HITL gate.
"""

from dataclasses import dataclass


@dataclass
class EscalationRecord:
    """One escalation iteration: the leak that triggered it and the fix applied."""

    iteration: int
    vector: str
    entity_type: str
    escalation: str


@dataclass
class LeakageVerdict:
    """Structured outcome of one adversarial probe over the sanitized context.

    ``vector`` is the attack strategy that produced the strongest signal,
    ``entity_type`` and ``evidence`` describe the residual identifier without
    echoing the raw value beyond what is needed to justify the verdict.
    """

    leaked: bool
    vector: str
    entity_type: str
    evidence: str
    confidence: float
