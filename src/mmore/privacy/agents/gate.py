"""Pre-cloud HITL approval gate.

Pipeline:  ... -> sanitizer -> leakage_adversary -> [gate] -> ...
Reads:     policy, risk, verdict, iteration, escalation_log
Writes:    summary, approved, outcome

The last step before the trust boundary: once the adversary clears the
sanitized context, the gate builds a concise, PII-free summary of everything
that was done and pauses for human approval via an interruption on the graph.
"""

import logging
from enum import Enum
from typing import List, Optional, Tuple, Union

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import interrupt
from typing_extensions import Self

from ...utils import load_config
from ..config import PrivacyConfig
from .base import BaseAgent
from .state import PreCloudOutcome, PrivacyState

logger = logging.getLogger(__name__)


class GateDecision(str, Enum):
    """The human's choice at the gate."""

    APPROVE = "approve"
    RETRY = "retry"
    REJECT = "reject"


_GATE_CHOICES: List[Tuple[GateDecision, str]] = [
    (GateDecision.APPROVE, "Approve: clear the sanitized context for the cloud call"),
    (GateDecision.RETRY, "Revise: tighten the policy and retry (optional feedback)"),
    (GateDecision.REJECT, "Reject: abort the request"),
]
_CHOICE_BY_NUMBER = {i + 1: decision for i, (decision, _) in enumerate(_GATE_CHOICES)}
_DECISION_BY_VALUE = {decision.value: decision for decision, _ in _GATE_CHOICES}


def _gate_options() -> List[dict]:
    """The numbered menu surfaced to the human in the interrupt payload."""
    return [
        {"choice": i + 1, "action": decision.value, "label": label}
        for i, (decision, label) in enumerate(_GATE_CHOICES)
    ]


def _as_choice_number(value: object) -> Optional[int]:
    """Read a menu number from an int or numeric string."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _interpret_decision(decision: object) -> Optional[GateDecision]:
    """Map a resume value (a menu number, an action name, or a dict) to a decision."""
    number = _as_choice_number(decision)
    if number is not None:
        return _CHOICE_BY_NUMBER.get(number)
    if isinstance(decision, str):
        return _DECISION_BY_VALUE.get(decision.strip().lower())
    if isinstance(decision, dict):
        return _interpret_decision(decision.get("choice"))
    return None


def _extract_feedback(decision: object) -> Optional[str]:
    """Pull the human's free-text guidance from a structured resume value."""
    if isinstance(decision, dict):
        value = decision.get("feedback")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_gate_summary(state: PrivacyState) -> str:
    """Build a concise, PII-free summary of the pre-cloud pipeline run."""
    policy = state.get("policy")
    risk = state.get("risk")
    verdict = state.get("verdict")
    iteration = state.get("iteration", 0)
    escalation_log = state.get("escalation_log") or []

    domain = policy.domain if policy else "unknown"
    strategy = policy.sanitization_strategy if policy else "unknown"

    if risk and risk.entity_counts:
        detection = ", ".join(
            f"{label}: {count}" for label, count in sorted(risk.entity_counts.items())
        )
    else:
        detection = "no sensitive entities detected"
    total = risk.count if risk else 0

    escalations = (
        ", ".join(r.escalation or "human feedback" for r in escalation_log)
        if escalation_log
        else "none"
    )

    if verdict is not None:
        probe = verdict.vector.value if verdict.vector else "none"
        gate_verdict = (
            f"adversary cleared the context (strongest probe {probe} "
            f"at confidence {verdict.confidence:.2f})"
        )
    else:
        gate_verdict = "adversary verdict unavailable"

    return "\n".join(
        [
            "Pre-cloud privacy review",
            f"- Domain: {domain}",
            f"- Detected (type: count): {detection}",
            f"- Total sensitive spans: {total}",
            f"- Sanitization strategy: {strategy}",
            f"- Escalation iterations: {iteration} ({escalations})",
            f"- Gate verdict: {gate_verdict}",
        ]
    )


class HITLGateAgent(BaseAgent):
    """Human approval gate at the pre-cloud trust boundary."""

    state_schema = PrivacyState
    node_name = "gate"

    def __init__(
        self,
        config: PrivacyConfig,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self._interactive = bool(config.interactive)
        super().__init__(config, llm_config=None, checkpointer=checkpointer)

    @classmethod
    def from_config(
        cls,
        config: Union[PrivacyConfig, str, dict],
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> Self:
        if not isinstance(config, PrivacyConfig):
            config = load_config(config, PrivacyConfig)
        return cls(config, checkpointer=checkpointer)

    def _node(self, state: PrivacyState) -> PrivacyState:
        """Build the summary and, when interactive, pause for human approval."""
        summary = build_gate_summary(state)
        if not self._interactive:
            return PrivacyState(
                summary=summary, approved=True, outcome=PreCloudOutcome.APPROVED
            )
        base = {"summary": summary, "options": _gate_options()}
        payload = base
        resume: object = None
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
                summary=summary, approved=True, outcome=PreCloudOutcome.APPROVED
            )
        if decision is GateDecision.REJECT:
            return PrivacyState(
                summary=summary, approved=False, outcome=PreCloudOutcome.REJECTED
            )

        # Else re-enter the privacy pipeline with Analyzer next
        return PrivacyState(
            summary=summary,
            approved=False,
            outcome=PreCloudOutcome.RE_LOOPED,
            human_feedback=_extract_feedback(resume),
        )
