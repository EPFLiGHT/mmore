"""The final report record returned alongside the answer.

A structured, PII-free, append-only audit record: one record per request. It
holds only types and counts, never raw information, so it can be persisted and
shown to the user.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .config import DetectionEngineType, SanitizationStrategyType
from .risk import RiskAssessment
from .verification import WarningKind


class PreCloudOutcome(str, Enum):
    """Outcome of a request at the pre-cloud trust boundary."""

    APPROVED = "approved"
    RE_LOOPED = "re-looped"
    ABORTED = "aborted"  # leak loop exhausted
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
    """One advisory warning kind and how many fired, no content."""

    kind: WarningKind
    count: int


@dataclass
class HITLEvent:
    """The pre-cloud HITL gate event."""

    fired: bool
    decision: Optional[HITLDecision] = None
    response_hash: Optional[str] = None


@dataclass
class ReportRecord:
    """The PII-free record emitted for one request."""

    request_id: str
    timestamp: str
    domain: str
    detection_engine: DetectionEngineType
    detection: RiskAssessment  # the detector's own span count + entity-type counts
    sanitization_strategy: SanitizationStrategyType
    escalation_iterations: int
    gate_outcome: PreCloudOutcome
    gate_iterations: int
    answer_backend: Optional[str]
    answer_model: Optional[str]
    advisory_warnings: List[WarningSummary]
    hitl: HITLEvent
    outcome: ReportOutcome
