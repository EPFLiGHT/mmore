"""Detector.

Pipeline:  analyzer  ->  [detector]  ->  sanitizer  ->  ... TODO: complete once implemented
Reads:     policy, raw_chunks
Writes:    spans, risk

One node in the privacy multi-agent pipeline: runs the policy's detection
engine over each raw chunk, deduplicates spans per chunk, and emits a coarse
risk assessment the next agents consume.
"""

import logging
from collections import Counter
from typing import Callable, Dict, List, Optional, Union

from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import Self

from ...utils import load_config
from ..config import PrivacyConfig
from ..detection.base import PIISpan
from ..detection.constants import DETECTION_TOOL_NAMES
from ..policy import PrivacyPolicy
from ..risk import RiskAssessment
from .base import BaseAgent
from .registry import tool_registry
from .state import PrivacyState

logger = logging.getLogger(__name__)


# ========================================================================
# Risk thresholds
# ========================================================================

# Density = total_spans / total_characters across all chunks
# TODO: Adjust after experiments
_RISK_DENSITY_MEDIUM = 0.005
_RISK_DENSITY_HIGH = 0.02


# ========================================================================
# Helpers
# ========================================================================


def _resolve_engine_tool(
    engine_short: str,
) -> Callable[[str, PrivacyPolicy], List[PIISpan]]:
    """Resolve the engine short name to a registered detection tool callable."""
    tool_name = DETECTION_TOOL_NAMES.get(engine_short)
    if tool_name is None:
        raise KeyError(
            f"Unknown detection engine '{engine_short}'. "
            f"Known engines: {sorted(DETECTION_TOOL_NAMES)}"
        )
    if tool_name not in tool_registry:
        raise KeyError(
            f"Detection tool '{tool_name}' is not registered. "
            f"Available tools: {sorted(tool_registry.keys())}"
        )
    return tool_registry[tool_name]


def _dedupe_spans(spans: List[PIISpan]) -> List[PIISpan]:
    """Collapse spans sharing (start, end, label) and keep the highest score."""
    best: Dict[tuple, PIISpan] = {}
    for span in spans:
        key = (span.start, span.end, span.label)
        prev = best.get(key)
        if prev is None or span.score > prev.score:
            best[key] = span
    return sorted(best.values(), key=lambda s: (s.start, s.end, s.label))


def _risk_level(density: float) -> str:
    if density >= _RISK_DENSITY_HIGH:
        return "high"
    if density >= _RISK_DENSITY_MEDIUM:
        return "medium"
    return "low"


def _build_risk(
    chunks: List[str], spans_per_chunk: List[List[PIISpan]]
) -> RiskAssessment:
    flat = [s for spans in spans_per_chunk for s in spans]
    entity_counts = dict(Counter(s.label for s in flat))
    total_chars = sum(len(c) for c in chunks)
    density = (len(flat) / total_chars) if total_chars else 0.0
    return RiskAssessment(
        count=len(flat),
        entity_counts=entity_counts,
        density=density,
        level=_risk_level(density),
    )


# ========================================================================
# Agent
# ========================================================================


class DetectorAgent(BaseAgent):
    """Runs the policy's detection engine over each raw chunk."""

    state_schema = PrivacyState
    node_name = "detector"

    def __init__(
        self,
        config: PrivacyConfig,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
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

    def detect(
        self, policy: PrivacyPolicy, chunks: List[str]
    ) -> tuple[List[List[PIISpan]], RiskAssessment]:
        """Run the policy's engine over each chunk and assess overall risk."""
        tool = _resolve_engine_tool(policy.detection_engine)

        spans_per_chunk: List[List[PIISpan]] = []
        for chunk in chunks:
            raw = tool(chunk, policy)
            spans_per_chunk.append(_dedupe_spans(raw))

        risk = _build_risk(chunks, spans_per_chunk)
        return spans_per_chunk, risk

    def _node(self, state: PrivacyState) -> PrivacyState:
        """Graph node: write spans and risk into the pipeline state."""
        policy = state.get("policy")
        if policy is None:
            raise ValueError("DetectorAgent requires 'policy' in the state.")
        chunks = list(state.get("raw_chunks", []))
        spans_per_chunk, risk = self.detect(policy, chunks)
        return PrivacyState(spans=spans_per_chunk, risk=risk)
