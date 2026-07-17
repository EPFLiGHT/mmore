"""Shared state for the privacy pipeline graph.

A single ``StateGraph(PrivacyState)`` flows through analyzer -> detector ->
sanitizer -> adversary -> HITL gate -> answer model -> verifier -> report.
Each agent contributes a node that reads what it needs and writes its
output back.
"""

from typing import Dict, List, Optional

from ..detection.base import PIISpan
from ..leakage import EscalationRecord, LeakageVerdict
from ..policy import PrivacyPolicy
from ..report import HITLEvent, PreCloudOutcome, ReportRecord
from ..risk import RiskAssessment
from ..verification import VerifierVerdict
from .base import NodeOutput


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
    query_spans: List[PIISpan]
    risk: Optional[RiskAssessment]
    sanitized_chunks: List[str]
    sanitized_query: str

    # Leakage adversary + escalation loop
    verdict: Optional[LeakageVerdict]
    safe: bool
    total_escalations: int  # all policy escalations (adversary + human)
    adversary_escalations: int
    escalation_log: List[EscalationRecord]
    skip_detection: bool

    # Pre-cloud HITL gate
    summary: str
    approved: Optional[bool]
    outcome: Optional[PreCloudOutcome]
    human_feedback: Optional[str]
    hitl_events: List[HITLEvent]

    # True when the revision comes from human, to let the Analyzer differentiate
    # with the ones from the adversary
    revision_requested: bool

    # Request metadata for the report
    request_id: str
    timestamp: str

    # Seconds each agent spent, summed over the escalation loop
    stage_seconds: Dict[str, float]

    # Post-cloud answer model
    answer: str
    answer_backend: Optional[str]  # backend: API provider or local
    answer_model: Optional[str]

    # Post-cloud advisory verifier
    verifier_verdict: Optional[VerifierVerdict]

    # Final append-only report records
    report: List[ReportRecord]
