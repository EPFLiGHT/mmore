"""The final report record returned alongside the answer.

A structured, PII-free, append-only audit record: one record per request. It
holds only types and counts, never raw information, so it can be persisted and
shown to the user.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

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
    """One verifier warning kind and how many fired."""

    kind: WarningKind
    count: int
    entity_type: Optional[str] = None
    confidence: float = 0.0


@dataclass
class HITLEvent:
    """A pre-cloud HITL gate event."""

    decision: HITLDecision
    human_feedback: Optional[str] = None


@dataclass
class ReportRecord:
    """The PII-free record emitted for one request."""

    request_id: str
    timestamp: str
    domain: str
    detection_engine: DetectionEngineType
    detection: RiskAssessment  # the detector's own span count + entity-type counts
    sanitization_strategy: SanitizationStrategyType
    adversary_iterations: int
    human_iterations: int
    gate_outcome: PreCloudOutcome
    answer_backend: Optional[str]
    answer_model: Optional[str]
    verifier_warnings: List[WarningSummary]
    verifier_checks_run: List[str]
    verifier_checks_failed: List[str]
    hitl_events: List[HITLEvent]
    outcome: ReportOutcome
    sanitized_query: str = ""
    stage_seconds: Dict[str, float] = field(default_factory=dict)
