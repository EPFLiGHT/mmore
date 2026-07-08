"""Builder for the final report."""

import json
from collections import Counter
from dataclasses import asdict
from typing import List, Optional

from .agents.state import PrivacyState
from .config import DetectionEngineType, SanitizationStrategyType
from .report import (
    HITLEvent,
    PreCloudOutcome,
    ReportOutcome,
    ReportRecord,
    WarningSummary,
)
from .risk import RiskAssessment
from .verification import VerifierVerdict

_ABORTED_OUTCOMES = (PreCloudOutcome.ABORTED, PreCloudOutcome.REJECTED)


def _warning_summaries(verdict: Optional[VerifierVerdict]) -> List[WarningSummary]:
    """Aggregate the advisory warnings to type + count, dropping any content."""
    counts = Counter(w.kind for w in verdict.warnings) if verdict else Counter()
    return [WarningSummary(kind=kind, count=count) for kind, count in counts.items()]


def _hitl_events(state: PrivacyState) -> List[HITLEvent]:
    """The human gate interactions recorded during the run (empty if none)."""
    return list(state.get("hitl_events") or [])


def _outcome(state: PrivacyState, verdict: Optional[VerifierVerdict]) -> ReportOutcome:
    if state.get("outcome") in _ABORTED_OUTCOMES or not state.get("answer"):
        return ReportOutcome.ABORTED_UNSAFE
    if verdict is not None and not verdict.clean:
        return ReportOutcome.RETURNED_WITH_WARNINGS
    return ReportOutcome.RETURNED


def build_report_record(state: PrivacyState) -> ReportRecord:
    """Build the PII-free record for one request from the final state."""
    policy = state.get("policy")
    gate_outcome = state.get("outcome")
    if policy is None or gate_outcome is None:
        raise ValueError("Report builder requires 'policy' and 'outcome' in the state.")

    verdict = state.get("verifier_verdict")
    return ReportRecord(
        request_id=state.get("request_id", ""),
        timestamp=state.get("timestamp", ""),
        domain=policy.domain,
        detection_engine=DetectionEngineType(policy.detection_engine),
        detection=state.get("risk") or RiskAssessment(count=0),
        sanitization_strategy=SanitizationStrategyType(policy.sanitization_strategy),
        adversary_iterations=state.get("adversary_escalations", 0),
        human_iterations=state.get("total_escalations", 0)
        - state.get("adversary_escalations", 0),
        gate_outcome=gate_outcome,
        answer_backend=state.get("answer_backend"),
        answer_model=state.get("answer_model"),
        advisory_warnings=_warning_summaries(verdict),
        hitl_events=_hitl_events(state),
        outcome=_outcome(state, verdict),
    )


def report_jsonl(record: ReportRecord) -> str:
    """Serialize a record as one append-only JSON line."""
    return json.dumps(asdict(record))
