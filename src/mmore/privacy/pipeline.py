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
from enum import Enum
from typing import Callable, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .agents.adversary import AdversarialAgent
from .agents.analyzer import ContextPolicyAnalyzerAgent
from .agents.base import NodeOutput
from .agents.detector import DetectorAgent
from .agents.sanitizer import SanitizerAgent
from .agents.state import PrivacyState
from .config import PrivacyConfig

logger = logging.getLogger(__name__)

# A pipeline node: reads the shared state and returns a partial update
NodeFn = Callable[..., NodeOutput]


class _Route(str, Enum):
    """Branches out of the adversary node in the pre-cloud loop."""

    PROCEED = "proceed"
    ESCALATE = "escalate"
    UNSAFE = "unsafe"


def _route_after_adversary(state: PrivacyState, max_iterations: int) -> _Route:
    """Decide branch once the adversary attacked the sanitized context."""
    if state.get("safe", False):
        return _Route.PROCEED
    if state.get("iteration", 0) >= max_iterations:
        return _Route.UNSAFE
    return _Route.ESCALATE


def _mark_unsafe_node(state: PrivacyState) -> PrivacyState:
    """Terminal for an exhausted loop: the request cannot reach the gate."""
    return PrivacyState(safe=False, outcome="aborted-unsafe")


def build_pipeline_graph(
    *,
    analyzer: NodeFn,
    detector: NodeFn,
    sanitizer: NodeFn,
    adversary: NodeFn,
    max_iterations: int = 3,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Compile the pre-cloud loop from explicit node callables.

    On a leak the adversary routes back to the analyzer, which escalates the
    policy. ``_PROCEED`` routes to ``END`` here; PR #4's HITL gate (and PR #5
    beyond it) replace that terminal edge with the gate node.
    """
    graph = StateGraph(PrivacyState)
    graph.add_node("analyzer", analyzer)
    graph.add_node("detector", detector)
    graph.add_node("sanitizer", sanitizer)
    graph.add_node("leakage_adversary", adversary)
    graph.add_node("mark_unsafe", _mark_unsafe_node)

    graph.add_edge(START, "analyzer")
    graph.add_edge("analyzer", "detector")
    graph.add_edge("detector", "sanitizer")
    graph.add_edge("sanitizer", "leakage_adversary")
    graph.add_conditional_edges(
        "leakage_adversary",
        lambda state: _route_after_adversary(state, max_iterations),
        {
            _Route.PROCEED: END,
            _Route.ESCALATE: "analyzer",
            _Route.UNSAFE: "mark_unsafe",
        },
    )
    graph.add_edge("mark_unsafe", END)
    return graph.compile(checkpointer=checkpointer)


def build_privacy_pipeline(
    config: PrivacyConfig,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Build the pre-cloud pipeline from a ``PrivacyConfig``.

    The agents provide the node callables; the compiled graph owns the single
    shared checkpointer (the agents are used only as node providers, so they
    are built without their own).
    """
    analyzer = ContextPolicyAnalyzerAgent.from_config(config)
    detector = DetectorAgent.from_config(config)
    sanitizer = SanitizerAgent.from_config(config)
    adversary = AdversarialAgent.from_config(config)
    return build_pipeline_graph(
        analyzer=analyzer.node,
        detector=detector.node,
        sanitizer=sanitizer.node,
        adversary=adversary.node,
        max_iterations=config.leakage_adversary.max_iterations,
        checkpointer=checkpointer,
    )
