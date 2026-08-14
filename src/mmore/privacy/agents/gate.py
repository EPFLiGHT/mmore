"""Pre-cloud HITL approval gate.

Pipeline:  ... -> sanitizer -> leakage_adversary -> [gate] -> ...
Reads:     policy, risk, verdict, total_escalations, escalation_log
Writes:    summary, approved, outcome, hitl_events, human_feedback

The last step before the trust boundary: once the adversary clears the
sanitized context, the gate builds a short, PII-free summary of everything
that was done and pauses for human approval via an interruption on the graph.
"""

import logging
from enum import Enum

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import interrupt
from typing_extensions import Self

from ...ux import plural
from ..config import PrivacyConfig, as_privacy_config
from ..schemas.leakage import SAFE_VERDICT
from ..schemas.report import HITLDecision, HITLEvent, PreCloudOutcome
from ..ux import tool_name
from .base import BaseAgent
from .state import PrivacyState

logger = logging.getLogger(__name__)

# What the resume value may look like: a menu number, an action name, or a
# {"choice": ..., "feedback": ...} mapping (hence the very loooong list below)
ResumeValue = dict[str, int | str | None] | int | str | None


class GateDecision(str, Enum):
    """The human's choice at the gate."""

    APPROVE = "approve"
    RETRY = "retry"
    REJECT = "reject"


_GATE_CHOICES: list[tuple[GateDecision, str]] = [
    (
        GateDecision.APPROVE,
        "Approve: call the answer model with safe query and context",
    ),
    (GateDecision.RETRY, "Revise: retry with optional feedback"),
    (GateDecision.REJECT, "Reject: abort the request"),
]
_CHOICE_BY_NUMBER = {i + 1: decision for i, (decision, _) in enumerate(_GATE_CHOICES)}


def _gate_options() -> list[dict[str, int | str]]:
    """The numbered menu surfaced to the human in the interrupt payload."""
    return [
        {"choice": i + 1, "action": decision.value, "label": label}
        for i, (decision, label) in enumerate(_GATE_CHOICES)
    ]


def _as_choice_number(value: int | str | None) -> int | None:
    """Read a menu number from an int or numeric string."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _interpret_decision(resume: ResumeValue) -> GateDecision | None:
    """Map a resume value to a decision, or None when it is unrecognized."""
    if isinstance(resume, dict):
        return _interpret_decision(resume.get("choice") or resume.get("action"))

    number = _as_choice_number(resume)
    if number is not None:
        return _CHOICE_BY_NUMBER.get(number)
    if isinstance(resume, str):
        try:
            return GateDecision(resume.strip().lower())
        except ValueError:
            return None
    return None


def _extract_feedback(resume: ResumeValue) -> str | None:
    """Pull the human's free-text guidance from a structured resume value."""
    if isinstance(resume, dict):
        value = resume.get("feedback")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _describe_adversary(state: PrivacyState) -> str:
    """How the leakage adversary's last probe went, in one line."""
    verdict = state.get("verdict")
    if verdict is None:
        return "not run"
    if verdict == SAFE_VERDICT:
        return "not run (disabled or no sanitized context)"
    if state.get("safe", False):
        return f"passed (nothing recovered, confidence {verdict.confidence:.2f})"
    probe = verdict.vector.value.replace("_", " ") if verdict.vector else "a probe"
    return (
        f"failed: {verdict.entity_type or 'sensitive data'} still recovered "
        f"via {probe} (confidence {verdict.confidence:.2f})"
    )


def build_gate_details(
    state: PrivacyState, max_iterations: int | None = None
) -> list[tuple[str, str]]:
    """Build a concise, PII-free summary of the pre-cloud pipeline run."""
    policy = state.get("policy")
    risk = state.get("risk")
    adversary_escalations = state.get("adversary_escalations", 0)
    human_revisions = state.get("total_escalations", 0) - adversary_escalations
    escalation_log = state.get("escalation_log") or []

    if risk and risk.entity_counts:
        detected = ", ".join(
            f"{label}: {count}" for label, count in sorted(risk.entity_counts.items())
        )
    else:
        detected = "nothing sensitive detected"

    if max_iterations is None:
        loops = f"{adversary_escalations} (adversary off)"
    else:
        escalations = (
            ", ".join(r.escalation or "human feedback" for r in escalation_log)
            if escalation_log
            else "none"
        )
        loops = f"{adversary_escalations}/{max_iterations} ({escalations})"

    return [
        ("Domain", str(policy.domain if policy else "unknown")),
        ("Detection engine", str(policy.detection_engine if policy else "unknown")),
        ("Detected (type: count)", detected),
        ("Total sensitive spans", str(risk.count if risk else 0)),
        (
            "Sanitization strategy",
            str(policy.sanitization_strategy if policy else "unknown"),
        ),
        ("Leakage escalations", loops),
        ("Human revisions", str(human_revisions)),
        ("Leak adversary", _describe_adversary(state)),
    ]


def build_gate_headline(state: PrivacyState, max_iterations: int | None = None) -> str:
    """The same review, condensed to a single line for a compact prompt."""
    policy = state.get("policy")
    risk = state.get("risk")
    verdict = state.get("verdict")
    escalations = state.get("adversary_escalations", 0)
    revisions = state.get("total_escalations", 0) - escalations

    parts: list[str] = []
    if policy:
        parts.append(
            f"{tool_name(policy.detection_engine)} (detector) + "
            f"{tool_name(policy.sanitization_strategy)} (sanitizer)"
        )
    if risk and risk.count:
        found = ", ".join(
            f"{label} {count}" for label, count in sorted(risk.entity_counts.items())
        )
        spans = plural(risk.count, "sensitive span")
        parts.append(f"{spans}: {found}" if found else spans)
    else:
        parts.append("nothing sensitive detected")
    if max_iterations is not None:
        parts.append(f"{escalations}/{max_iterations} leak escalations")
    if revisions:
        parts.append(plural(revisions, "revision"))
    if verdict is not None and not state.get("safe", False):
        parts.append(f"adversary recovered {verdict.entity_type or 'sensitive data'}")
    return " | ".join(parts)


def build_gate_summary(state: PrivacyState, max_iterations: int | None = None) -> str:
    """Build a concise, PII-free summary of the pre-cloud pipeline run."""
    lines = [
        f"- {label}: {value}"
        for label, value in build_gate_details(state, max_iterations)
    ]
    return "\n".join(["Pre-cloud privacy review", *lines])


class HITLGateAgent(BaseAgent):
    """Human approval gate at the pre-cloud trust boundary."""

    state_schema = PrivacyState
    node_name = "gate"

    def __init__(
        self,
        config: PrivacyConfig,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self._interactive = config.interactive
        adversary = config.leakage_adversary
        self._max_iterations = adversary.max_iterations if adversary.enabled else None
        super().__init__(config, llm_config=None, checkpointer=checkpointer)

    @classmethod
    def from_config(
        cls,
        config: PrivacyConfig | str,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> Self:
        return cls(as_privacy_config(config), checkpointer=checkpointer)

    def _review_payload(self, state: PrivacyState) -> dict:
        """Everything the human needs to decide, PII-free except the diff itself."""
        return {
            "summary": build_gate_summary(state, self._max_iterations),
            "headline": build_gate_headline(state, self._max_iterations),
            "details": [
                {"label": label, "value": value}
                for label, value in build_gate_details(state, self._max_iterations)
            ],
            "options": _gate_options(),
            "query": {
                "raw": state.get("query", ""),
                "sanitized": state.get("sanitized_query", ""),
            },
            "chunks": [
                {"raw": raw, "sanitized": sanitized}
                for raw, sanitized in zip(
                    state.get("raw_chunks", []), state.get("sanitized_chunks", [])
                )
            ],
        }

    def _node(self, state: PrivacyState) -> PrivacyState:
        """Build the summary and, when interactive, pause for human approval."""
        summary = build_gate_summary(state, self._max_iterations)
        if not self._interactive:
            return PrivacyState(
                summary=summary, approved=True, outcome=PreCloudOutcome.APPROVED
            )

        review = self._review_payload(state)
        payload = review
        resume: ResumeValue = None
        decision = None
        while decision is None:  # re-prompt until the human gives a valid choice
            resume = interrupt(payload)
            decision = _interpret_decision(resume)
            payload = {
                **review,
                "error": "Unrecognized choice: reply with 1, 2, or 3.",
            }

        match decision:
            case GateDecision.APPROVE:
                return PrivacyState(
                    summary=summary,
                    approved=True,
                    outcome=PreCloudOutcome.APPROVED,
                    hitl_events=self._record(state, HITLDecision.APPROVE),
                )
            case GateDecision.REJECT:
                return PrivacyState(
                    summary=summary,
                    approved=False,
                    outcome=PreCloudOutcome.REJECTED,
                    hitl_events=self._record(state, HITLDecision.REJECT),
                )
            case _:  # GateDecision.RETRY
                # Re-enter the privacy pipeline with the Analyzer next
                feedback = _extract_feedback(resume)
                return PrivacyState(
                    summary=summary,
                    approved=False,
                    outcome=PreCloudOutcome.RE_LOOPED,
                    human_feedback=feedback,
                    revision_requested=True,
                    hitl_events=self._record(state, HITLDecision.RETRY, feedback),
                )

    @staticmethod
    def _record(
        state: PrivacyState, decision: HITLDecision, feedback: str | None = None
    ) -> list[HITLEvent]:
        """Append one human gate decision to the run's accumulated event list."""
        event = HITLEvent(decision=decision, human_feedback=feedback)
        return state.get("hitl_events", []) + [event]
