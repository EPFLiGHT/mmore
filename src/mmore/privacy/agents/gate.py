"""Pre-cloud HITL approval gate.

Pipeline:  ... -> sanitizer -> leakage_adversary -> [gate] -> ...
Reads:     policy, risk, verdict, total_escalations, escalation_log
Writes:    summary, approved, outcome, hitl_events, human_feedback

The last step before the trust boundary: once the adversary clears the
sanitized context, the gate builds a concise, PII-free summary of everything
that was done and pauses for human approval via an interruption on the graph.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import interrupt
from typing_extensions import Self

from ...utils import load_config
from ...ux import plural
from ..config import PrivacyConfig
from ..leakage import SAFE_VERDICT
from ..report import HITLDecision, HITLEvent
from ..ux import tool_name
from .base import BaseAgent
from .state import PreCloudOutcome, PrivacyState

logger = logging.getLogger(__name__)


def _appended_events(
    state: PrivacyState, decision: HITLDecision, feedback: Optional[str] = None
) -> List[HITLEvent]:
    """Append one human gate decision to the run's accumulated event list."""
    event = HITLEvent(decision=decision, human_feedback=feedback)
    return list(state.get("hitl_events", [])) + [event]


class GateDecision(str, Enum):
    """The human's choice at the gate."""

    APPROVE = "approve"
    RETRY = "retry"
    REJECT = "reject"


_GATE_CHOICES: List[Tuple[GateDecision, str]] = [
    (
        GateDecision.APPROVE,
        "Approve: call the answer model with safe query and context",
    ),
    (GateDecision.RETRY, "Revise: retry with optional feedback"),
    (GateDecision.REJECT, "Reject: abort the request"),
]
_CHOICE_BY_NUMBER = {i + 1: decision for i, (decision, _) in enumerate(_GATE_CHOICES)}


def _gate_options() -> List[Dict[str, int | str]]:
    """The numbered menu surfaced to the human in the interrupt payload."""
    return [
        {"choice": i + 1, "action": decision.value, "label": label}
        for i, (decision, label) in enumerate(_GATE_CHOICES)
    ]


def _as_choice_number(value: int | str | None) -> Optional[int]:
    """Read a menu number from an int or numeric string."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _interpret_decision(
    decision: Dict[str, int | str | None] | int | str | None,
) -> Optional[GateDecision]:
    """Map a resume value (a menu number, an action name, or a dict) to a decision."""
    if isinstance(decision, dict):
        return _interpret_decision(decision.get("choice") or decision.get("action"))

    number = _as_choice_number(decision)
    if number is not None:
        return _CHOICE_BY_NUMBER.get(number)
    if isinstance(decision, str):
        try:
            return GateDecision(decision.strip().lower())
        except ValueError:
            return None
    return None


def _extract_feedback(
    decision: Dict[str, int | str | None] | int | str | None,
) -> Optional[str]:
    """Pull the human's free-text guidance from a structured resume value."""
    if isinstance(decision, dict):
        value = decision.get("feedback")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_gate_details(
    state: PrivacyState, max_iterations: Optional[int] = None
) -> List[Tuple[str, str]]:
    """Build a concise, PII-free summary of the pre-cloud pipeline run."""

    policy = state.get("policy")
    risk = state.get("risk")
    verdict = state.get("verdict")
    adversary_escalations = state.get("adversary_escalations", 0)
    human_revisions = state.get("total_escalations", 0) - adversary_escalations
    escalation_log = state.get("escalation_log") or []

    if risk and risk.entity_counts:
        detected = ", ".join(
            f"{label}: {count}" for label, count in sorted(risk.entity_counts.items())
        )
    else:
        detected = "nothing sensitive detected"

    escalations = (
        ", ".join(r.escalation or "human feedback" for r in escalation_log)
        if escalation_log
        else "none"
    )

    if verdict is None:
        gate_verdict = "not run"
    elif verdict == SAFE_VERDICT:
        gate_verdict = "not run (disabled or no sanitized context)"
    elif state.get("safe", False):
        gate_verdict = (
            f"passed (nothing recovered, confidence {verdict.confidence:.2f})"
        )
    else:
        probe = verdict.vector.value.replace("_", " ") if verdict.vector else "a probe"
        leaked = verdict.entity_type or "sensitive data"
        gate_verdict = (
            f"failed: {leaked} still recovered via {probe} "
            f"(confidence {verdict.confidence:.2f})"
        )

    if max_iterations is None:
        loops = f"{adversary_escalations} (adversary off)"
    else:
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
        ("Leak adversary", gate_verdict),
    ]


def build_gate_headline(
    state: PrivacyState, max_iterations: Optional[int] = None
) -> str:
    policy = state.get("policy")
    risk = state.get("risk")
    verdict = state.get("verdict")
    escalations = state.get("adversary_escalations", 0)
    revisions = state.get("total_escalations", 0) - escalations

    parts: List[str] = []
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


def build_gate_summary(
    state: PrivacyState, max_iterations: Optional[int] = None
) -> str:
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
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self._interactive = config.interactive
        adversary = config.leakage_adversary
        self._max_iterations = adversary.max_iterations if adversary.enabled else None
        super().__init__(config, llm_config=None, checkpointer=checkpointer)

    @classmethod
    def from_config(
        cls,
        config: PrivacyConfig | str,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> Self:
        if not isinstance(config, PrivacyConfig):
            config = load_config(config, PrivacyConfig)
        return cls(config, checkpointer=checkpointer)

    def _node(self, state: PrivacyState) -> PrivacyState:
        """Build the summary and, when interactive, pause for human approval."""
        summary = build_gate_summary(state, self._max_iterations)
        if not self._interactive:
            return PrivacyState(
                summary=summary, approved=True, outcome=PreCloudOutcome.APPROVED
            )
        base = {
            "summary": summary,
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
        payload = base
        resume = None
        decision = None
        while decision is None:  # re-prompt until the human gives a valid choice
            resume = interrupt(payload)
            decision = _interpret_decision(resume)
            if decision is None:
                payload = {
                    **base,
                    "error": "Unrecognized choice: reply with 1, 2, or 3.",
                }
        if decision is GateDecision.APPROVE:
            return PrivacyState(
                summary=summary,
                approved=True,
                outcome=PreCloudOutcome.APPROVED,
                hitl_events=_appended_events(state, HITLDecision.APPROVE),
            )
        if decision is GateDecision.REJECT:
            return PrivacyState(
                summary=summary,
                approved=False,
                outcome=PreCloudOutcome.REJECTED,
                hitl_events=_appended_events(state, HITLDecision.REJECT),
            )

        # Else re-enter the privacy pipeline with Analyzer next
        feedback = _extract_feedback(resume)
        return PrivacyState(
            summary=summary,
            approved=False,
            outcome=PreCloudOutcome.RE_LOOPED,
            human_feedback=feedback,
            revision_requested=True,
            hitl_events=_appended_events(state, HITLDecision.RETRY, feedback),
        )
