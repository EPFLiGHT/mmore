"""Builder for the final report."""

from .agents.state import PrivacyState
from .config import DetectionEngineType, SanitizationStrategyType, VerifierCheck
from .schemas.report import (
    PreCloudOutcome,
    ReportOutcome,
    ReportRecord,
    WarningSummary,
)
from .schemas.risk import RiskAssessment
from .schemas.verification import VerifierVerdict, VerifierWarning

_ABORTED_OUTCOMES = (PreCloudOutcome.ABORTED, PreCloudOutcome.REJECTED)


def _warning_summaries(verdict: VerifierVerdict | None) -> list[WarningSummary]:
    """Aggregate the verifier warnings to type + count, dropping any content."""
    if verdict is None:
        return []
    groups: dict[tuple[VerifierCheck, str | None], list[VerifierWarning]] = {}
    for warning in verdict.warnings:
        by_entity = warning.kind is VerifierCheck.RESIDUAL_LEAKAGE
        key = (warning.kind, warning.flagged if by_entity else None)
        groups.setdefault(key, []).append(warning)

    return [
        WarningSummary(
            kind=kind,
            count=len(group),
            entity_type=entity_type,
            confidence=max(w.confidence for w in group),
        )
        for (kind, entity_type), group in groups.items()
    ]


def _outcome(state: PrivacyState, verdict: VerifierVerdict | None) -> ReportOutcome:
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
    total_escalations = state.get("total_escalations", 0)
    adversary_escalations = state.get("adversary_escalations", 0)
    return ReportRecord(
        request_id=state.get("request_id", ""),
        timestamp=state.get("timestamp", ""),
        domain=policy.domain,
        detection_engine=DetectionEngineType(policy.detection_engine),
        detection=state.get("risk") or RiskAssessment(count=0),
        sanitization_strategy=SanitizationStrategyType(policy.sanitization_strategy),
        adversary_iterations=adversary_escalations,
        human_iterations=total_escalations - adversary_escalations,
        gate_outcome=gate_outcome,
        answer_backend=state.get("answer_backend"),
        answer_model=state.get("answer_model"),
        verifier_warnings=_warning_summaries(verdict),
        verifier_checks_run=list(verdict.checks_run) if verdict else [],
        verifier_checks_failed=list(verdict.checks_failed) if verdict else [],
        hitl_events=list(state.get("hitl_events") or []),
        outcome=_outcome(state, verdict),
        sanitized_query=state.get("sanitized_query", ""),
        stage_seconds=dict(state.get("stage_seconds", {})),
    )
