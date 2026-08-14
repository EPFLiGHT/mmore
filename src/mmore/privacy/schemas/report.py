"""The final report record returned alongside the answer.

A structured, PII-free, append-only audit record: one record per request. It
holds only types and counts, never raw information, so it can be persisted and
shown to the user.
"""

from dataclasses import dataclass, field
from enum import Enum

from ..config import DetectionEngineType, SanitizationStrategyType, VerifierCheck
from .risk import RiskAssessment


class PreCloudOutcome(str, Enum):
    """Outcome of a request at the pre-cloud trust boundary."""

    APPROVED = "approved"
    RE_LOOPED = "re-looped"
    ABORTED = "aborted"  # it means leak loop exhausted
    REJECTED = "rejected"


class ReportOutcome(str, Enum):
    """How the request ended."""

    RETURNED = "returned"
    RETURNED_WITH_WARNINGS = "returned-with-warnings"
    ABORTED_UNSAFE = "aborted-unsafe"


class HITLDecision(str, Enum):
    """The human's recorded decision at the pre-cloud gate."""

    APPROVE = "approve"
    RETRY = "retry"
    REJECT = "reject"


@dataclass
class WarningSummary:
    """One verifier warning kind and how many fired."""

    kind: VerifierCheck
    count: int
    entity_type: str | None = None
    confidence: float = 0.0


@dataclass
class HITLEvent:
    """A pre-cloud HITL gate event."""

    decision: HITLDecision
    human_feedback: str | None = None


@dataclass
class ReportRecord:
    """The PII-free record emitted for one request."""

    request_id: str
    timestamp: str
    domain: str
    detection_engine: DetectionEngineType
    detection: RiskAssessment
    sanitization_strategy: SanitizationStrategyType
    adversary_iterations: int
    human_iterations: int
    gate_outcome: PreCloudOutcome
    answer_backend: str | None
    answer_model: str | None
    verifier_warnings: list[WarningSummary]
    verifier_checks_run: list[str]
    verifier_checks_failed: list[str]
    hitl_events: list[HITLEvent]
    outcome: ReportOutcome
    sanitized_query: str = ""
    stage_seconds: dict[str, float] = field(default_factory=dict)
