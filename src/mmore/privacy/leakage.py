"""The verdict produced by the pre-cloud Leakage Adversary.

Emitted by the AdversarialAgent after probing the sanitized context
for residual PII and quasi-identifiers, and consumed by the escalation loop
and the HITL gate.
"""

from dataclasses import dataclass
from typing import Optional

from .config import AttackVector


@dataclass
class EscalationRecord:
    """One escalation iteration: what triggered it and the fix applied."""

    iteration: int
    escalation: Optional[str] = None
    from_human_feedback: bool = False
    vector: Optional[AttackVector] = None
    entity_type: Optional[str] = None


@dataclass
class LeakageVerdict:
    """Structured outcome of one adversarial probe over the sanitized context."""

    leaked: bool
    vector: Optional[AttackVector]
    entity_type: Optional[str]
    evidence: str
    confidence: float


# Verdict for a context with nothing to attack
SAFE_VERDICT = LeakageVerdict(
    leaked=False, vector=None, entity_type=None, evidence="", confidence=0.0
)
