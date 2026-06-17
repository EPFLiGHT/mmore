"""Shared state for the privacy pipeline graph.

A single ``StateGraph(PrivacyState)`` flows through analyzer -> detector ->
sanitizer -> adversary -> HITL gate -> answer model -> verifier -> report.
Each agent contributes a node that reads what it needs and writes its
output back.
"""

from enum import Enum
from typing import List, Optional

from ..detection.base import PIISpan
from ..leakage import EscalationRecord, LeakageVerdict
from ..policy import PrivacyPolicy
from ..risk import RiskAssessment
from ..verification import VerifierVerdict
from .base import NodeOutput


class PreCloudOutcome(str, Enum):
    """Outcome of a request at the pre-cloud trust boundary."""

    APPROVED = "approved"
    RE_LOOPED = "re-looped"
    ABORTED = "aborted"  # leak loop exhausted
    REJECTED = "rejected"


class PrivacyState(NodeOutput, total=False):
    """Pipeline state and node output for the privacy graph.

    A privacy-specific node output: every agent node returns a partial
    ``PrivacyState``, writing only the fields it produces. ``query`` and
    ``raw_chunks`` are populated by the caller before the graph runs.
    """

    query: str
    raw_chunks: List[str]
    policy: Optional[PrivacyPolicy]
    spans: List[List[PIISpan]]
    risk: Optional[RiskAssessment]
    sanitized_chunks: List[str]

    # Leakage adversary + escalation loop
    verdict: Optional[LeakageVerdict]
    safe: bool
    iteration: int  # total escalations
    leak_iterations: int  # leak-driven escalations only
    escalation_log: List[EscalationRecord]

    # Pre-cloud HITL gate
    summary: str
    approved: Optional[bool]
    outcome: Optional[PreCloudOutcome]
    human_feedback: Optional[str]

    # Request metadata for the report
    request_id: str
    timestamp: str

    # Post-cloud answer model
    answer: str
    answer_backend: Optional[str]  # backend: API provider or local
    answer_model: Optional[str]

    # Post-cloud advisory verifier
    verifier_verdict: Optional[VerifierVerdict]

    # Final append-only report records
    report: List[dict]  # TODO: have a report record here
