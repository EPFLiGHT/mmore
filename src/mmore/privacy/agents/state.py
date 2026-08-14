"""Shared state for the privacy pipeline graph.

A single ``StateGraph(PrivacyState)`` flows through analyzer -> detector ->
sanitizer -> adversary -> HITL gate -> answer model -> verifier -> report.
Each agent contributes a node that reads what it needs and writes its
output back.
"""

from ..detection.base import PIISpan
from ..schemas.leakage import EscalationRecord, LeakageVerdict
from ..schemas.policy import PrivacyPolicy
from ..schemas.report import HITLEvent, PreCloudOutcome, ReportRecord
from ..schemas.risk import RiskAssessment
from ..schemas.verification import VerifierVerdict
from .base import NodeOutput


class PrivacyState(NodeOutput, total=False):
    """Pipeline state and node output for the privacy graph.

    A privacy-specific node output: every agent node returns a partial
    ``PrivacyState``, writing only the fields it produces. ``query`` and
    ``raw_chunks`` are populated by the caller before the graph runs.
    """

    query: str
    raw_chunks: list[str]
    policy: PrivacyPolicy | None
    spans: list[list[PIISpan]]
    query_spans: list[PIISpan]
    risk: RiskAssessment | None
    sanitized_chunks: list[str]
    sanitized_query: str

    # Leakage adversary + escalation loop
    verdict: LeakageVerdict | None
    safe: bool
    total_escalations: int  # all policy escalations (adversary + human)
    adversary_escalations: int
    escalation_log: list[EscalationRecord]
    skip_detection: bool

    # Pre-cloud HITL gate
    summary: str
    approved: bool | None
    outcome: PreCloudOutcome | None
    human_feedback: str | None
    hitl_events: list[HITLEvent]

    # True when the revision comes from human, to let the Analyzer differentiate
    # with the ones from the adversary
    revision_requested: bool

    # Request metadata for the report
    request_id: str
    timestamp: str

    # Seconds each agent spent, summed over the escalation loop
    stage_seconds: dict[str, float]

    # Post-cloud answer model
    answer: str
    answer_backend: str | None  # backend: API provider or local
    answer_model: str | None

    # Post-cloud advisory verifier
    verifier_verdict: VerifierVerdict | None

    # Final append-only report records
    report: list[ReportRecord]
