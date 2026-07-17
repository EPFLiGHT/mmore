"""Pre-cloud privacy pipeline graph.

Wires the graph that runs the chain

    analyzer -> detector -> sanitizer -> leakage_adversary -> (HITL gate)

with a bounded escalation loop. The analyzer is the single policy authority:
when the adversary flags a leak and the iteration budget is not spent, the
graph loops back to the analyzer, which escalates the policy before the chain
re-detects and re-sanitizes. On exhaustion the request is marked unsafe and
cannot proceed.
"""

import logging
import time
from dataclasses import replace
from enum import Enum
from typing import Optional, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .agents.adversary import AdversarialAgent
from .agents.analyzer import ContextPolicyAnalyzerAgent
from .agents.detector import DetectorAgent
from .agents.gate import HITLGateAgent
from .agents.sanitizer import SanitizerAgent
from .agents.state import PreCloudOutcome, PrivacyState
from .agents.verifier import AdvisoryVerifierAgent
from .answer import AnswerModel
from .config import PrivacyConfig
from .report_builder import build_report_record
from .ux import ANSWER_STAGE, Stage, report_stage

logger = logging.getLogger(__name__)


# A pipeline node: reads the shared state and returns a partial PrivacyState
class NodeFn(Protocol):
    def __call__(self, state: PrivacyState) -> PrivacyState: ...


class _Node(str, Enum):
    """Graph node ids across the full pipeline."""

    ANALYZER = "analyzer"
    DETECTOR = "detector"
    SANITIZER = "sanitizer"
    ADVERSARY = "leakage_adversary"
    GATE = "gate"
    MARK_UNSAFE = "mark_unsafe"
    ANSWER = "answer"
    VERIFIER = "advisory_verifier"
    REPORT = "report"


class _Route(str, Enum):
    """Branches out of the adversary and gate nodes in the pre-cloud loop."""

    PROCEED = "proceed"
    ESCALATE = "escalate"
    UNSAFE = "unsafe"
    REJECTED = "rejected"


# What the user sees while each agent runs
_STAGES: dict[str, Stage] = {
    _Node.ANALYZER: Stage(
        "Analyzer", "reading the context and setting the privacy policy", "analyzing"
    ),
    _Node.DETECTOR: Stage(
        "Detector", "scanning the query and the chunks for sensitive data", "detecting"
    ),
    _Node.SANITIZER: Stage(
        "Sanitizer", "masking or rewriting what was flagged", "sanitizing"
    ),
    _Node.ADVERSARY: Stage(
        "Adversary", "attacking the sanitized context to find leaks", "probing"
    ),
    _Node.GATE: Stage(
        "Gate", "waiting for your approval before the answer model call", "reviewing"
    ),
    _Node.ANSWER: Stage(
        ANSWER_STAGE, "answering from the sanitized context only", "answering"
    ),
    _Node.VERIFIER: Stage(
        "Verifier", "checking the answer for leaks and faithfulness", "verifying"
    ),
}


def _with_tool(stage: Stage, node_id: str, state: PrivacyState) -> Stage:
    """Name the tool the policy picked, for the agents that run one."""
    policy = state.get("policy")
    if policy is None:
        return stage
    if node_id == _Node.DETECTOR:
        return replace(stage, unit=f"{stage.unit} w/ {policy.detection_engine}")
    if node_id == _Node.SANITIZER:
        return replace(stage, unit=f"{stage.unit} w/ {policy.sanitization_strategy}")
    return stage


def _staged(node: NodeFn, node_id: str, stage: Stage) -> NodeFn:
    """Announce the agent before running its node, and record what it cost."""

    def run(state: PrivacyState) -> PrivacyState:
        report_stage(_with_tool(stage, node_id, state))
        start = time.perf_counter()
        result = node(state)
        seconds = dict(state.get("stage_seconds", {}))
        elapsed = time.perf_counter() - start
        seconds[stage.agent] = seconds.get(stage.agent, 0.0) + elapsed
        result["stage_seconds"] = seconds
        return result

    return run


def _route_after_adversary(
    state: PrivacyState, max_iterations: int, abort_on_exhaustion: bool
) -> _Route:
    """Decide branch once the adversary attacked the sanitized context."""
    if state.get("safe", False):
        return _Route.PROCEED
    if state.get("adversary_escalations", 0) >= max_iterations:
        return _Route.UNSAFE if abort_on_exhaustion else _Route.PROCEED
    return _Route.ESCALATE


def _route_after_gate(state: PrivacyState) -> _Route:
    """Decide branch once the gate has recorded the human's decision."""
    if state.get("approved", False):
        return _Route.PROCEED
    if state.get("outcome") == PreCloudOutcome.REJECTED:
        return _Route.REJECTED
    return _Route.ESCALATE


def _mark_unsafe_node(state: PrivacyState) -> PrivacyState:
    """Terminal for an exhausted loop: the request cannot reach the gate."""
    return PrivacyState(safe=False, outcome=PreCloudOutcome.ABORTED)


def _report_node(state: PrivacyState) -> PrivacyState:
    """Terminal node: append this request's PII-free report record."""
    record = build_report_record(state)
    return PrivacyState(report=[*state.get("report", []), record])


def build_pipeline_graph(
    *,
    analyzer: NodeFn,
    detector: NodeFn,
    sanitizer: NodeFn,
    adversary: NodeFn,
    gate: NodeFn,
    answer: NodeFn,
    verifier: NodeFn,
    max_iterations: int = 3,
    abort_on_exhaustion: bool = True,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Compile the full pipeline from explicit node callables."""
    graph = StateGraph(PrivacyState)
    for node_id, node in (
        (_Node.ANALYZER, analyzer),
        (_Node.DETECTOR, detector),
        (_Node.SANITIZER, sanitizer),
        (_Node.ADVERSARY, adversary),
        (_Node.GATE, gate),
        (_Node.ANSWER, answer),
        (_Node.VERIFIER, verifier),
    ):
        graph.add_node(node_id, _staged(node, node_id, _STAGES[node_id]))
    graph.add_node(_Node.MARK_UNSAFE, _mark_unsafe_node)
    graph.add_node(_Node.REPORT, _report_node)

    graph.add_edge(START, _Node.ANALYZER)
    graph.add_edge(_Node.ANALYZER, _Node.DETECTOR)
    graph.add_edge(_Node.DETECTOR, _Node.SANITIZER)
    graph.add_edge(_Node.SANITIZER, _Node.ADVERSARY)
    graph.add_conditional_edges(
        _Node.ADVERSARY,
        lambda state: _route_after_adversary(
            state, max_iterations, abort_on_exhaustion
        ),
        {
            _Route.PROCEED: _Node.GATE,
            _Route.ESCALATE: _Node.ANALYZER,
            _Route.UNSAFE: _Node.MARK_UNSAFE,
        },
    )
    graph.add_conditional_edges(
        _Node.GATE,
        _route_after_gate,
        {
            _Route.PROCEED: _Node.ANSWER,
            _Route.REJECTED: _Node.REPORT,
            _Route.ESCALATE: _Node.ANALYZER,
        },
    )
    graph.add_edge(_Node.MARK_UNSAFE, _Node.REPORT)
    graph.add_edge(_Node.ANSWER, _Node.VERIFIER)
    graph.add_edge(_Node.VERIFIER, _Node.REPORT)
    graph.add_edge(_Node.REPORT, END)
    return graph.compile(checkpointer=checkpointer)


def build_privacy_pipeline(
    config: PrivacyConfig,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Build the full privacy pipeline from a ``PrivacyConfig``.

    The agents provide the node callables: the compiled graph owns the single
    shared checkpointer (the agents are used only as node providers, so they
    are built without their own).
    """
    analyzer = ContextPolicyAnalyzerAgent.from_config(config)
    detector = DetectorAgent.from_config(config)
    sanitizer = SanitizerAgent.from_config(config)
    adversary = AdversarialAgent.from_config(config)
    gate = HITLGateAgent.from_config(config)
    answer = AnswerModel.from_config(config)
    verifier = AdvisoryVerifierAgent.from_config(config)

    return build_pipeline_graph(
        analyzer=analyzer._node,
        detector=detector._node,
        sanitizer=sanitizer._node,
        adversary=adversary._node,
        gate=gate._node,
        answer=answer._node,
        verifier=verifier._node,
        max_iterations=config.leakage_adversary.max_iterations,
        abort_on_exhaustion=config.leakage_adversary.abort_on_exhaustion,
        checkpointer=checkpointer,
    )
