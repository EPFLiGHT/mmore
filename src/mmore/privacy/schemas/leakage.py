"""The verdict produced by the pre-cloud Leakage Adversary.

Emitted by the AdversarialAgent after probing the sanitized context
for residual PII and quasi-identifiers, and used by the escalation loop
and the HITL gate.
"""

from dataclasses import dataclass

from ..config import AttackVector


@dataclass
class EscalationRecord:
    """One escalation iteration: what triggered it and the fix applied."""

    iteration: int
    escalation: str | None = None
    from_human_feedback: bool = False
    vector: AttackVector | None = None
    entity_type: str | None = None
    report: str | None = None  # the guidance text for the escalation


@dataclass
class LeakageVerdict:
    """Structured outcome of one adversarial probe over the sanitized context."""

    leaked: bool
    vector: AttackVector | None
    entity_type: str | None
    evidence: str
    confidence: float
    recommendation: str | None = None


# Verdict for a context with nothing to attack
SAFE_VERDICT = LeakageVerdict(
    leaked=False, vector=None, entity_type=None, evidence="", confidence=0.0
)
