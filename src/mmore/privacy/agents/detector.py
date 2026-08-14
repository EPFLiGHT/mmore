"""Detector.

Pipeline:  analyzer  ->  [detector]  ->  sanitizer  ->  adversarial -> ...
Reads:     policy, raw_chunks
Writes:    spans, risk

One node in the privacy multi-agent pipeline: runs the policy's detection
engine over each raw chunk, deduplicates spans per chunk, and emits a coarse
risk assessment the next agents consume.
"""

import logging
from collections import Counter

from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import Self

from ...rag.llm import LLMConfig
from ..config import PrivacyConfig, as_privacy_config
from ..detection.base import DetectionTool, PIISpan
from ..detection.constants import DETECTION_TOOL_NAMES
from ..detection.llm_engine import LLMDetectionEngine
from ..schemas.policy import PrivacyPolicy
from ..schemas.risk import RiskAssessment
from .base import BaseAgent
from .registry import ToolNotRegisteredError, resolve_tool
from .state import PrivacyState

logger = logging.getLogger(__name__)

# Density = total_spans / total_characters across all chunks
_RISK_DENSITY_MEDIUM = 0.005
_RISK_DENSITY_HIGH = 0.02


def _resolve_engine_tool(engine: str) -> DetectionTool:
    """Resolve an engine short name to its registered detection tool."""
    tool_name = DETECTION_TOOL_NAMES.get(engine)
    if tool_name is None:
        raise ToolNotRegisteredError(
            f"Unknown detection engine '{engine}'. "
            f"Known engines: {sorted(DETECTION_TOOL_NAMES)}"
        )
    return resolve_tool(tool_name)


def _dedupe_spans(spans: list[PIISpan]) -> list[PIISpan]:
    """Collapse spans sharing (start, end, label) and keep the highest score."""
    best: dict[tuple[int, int, str], PIISpan] = {}
    for span in spans:
        key = (span.start, span.end, span.label)
        if key not in best or span.score > best[key].score:
            best[key] = span
    return sorted(best.values(), key=lambda s: (s.start, s.end, s.label))


def _assess_risk(
    chunks: list[str], spans_per_chunk: list[list[PIISpan]]
) -> RiskAssessment:
    spans = [span for chunk_spans in spans_per_chunk for span in chunk_spans]
    total_chars = sum(len(chunk) for chunk in chunks)
    density = len(spans) / total_chars if total_chars else 0.0
    if density >= _RISK_DENSITY_HIGH:
        level = "high"
    elif density >= _RISK_DENSITY_MEDIUM:
        level = "medium"
    else:
        level = "low"
    return RiskAssessment(
        count=len(spans),
        entity_counts=dict(Counter(span.label for span in spans)),
        density=density,
        level=level,
    )


class DetectorAgent(BaseAgent):
    """Runs the policy's detection engine over each raw chunk."""

    state_schema = PrivacyState
    node_name = "detector"

    def __init__(
        self,
        config: PrivacyConfig,
        llm_config: LLMConfig | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        super().__init__(config, llm_config=llm_config, checkpointer=checkpointer)

    @classmethod
    def from_config(
        cls,
        config: PrivacyConfig | str,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> Self:
        config = as_privacy_config(config)
        llm_config = config.detection.llm
        if llm_config is None and config.context_analyzer:
            llm_config = config.context_analyzer.llm
        return cls(config, llm_config, checkpointer=checkpointer)

    def _resolve_tool(self, policy: PrivacyPolicy) -> DetectionTool:
        """The engine callable for ``policy`` according the agent's own LLM."""
        if policy.detection_engine == "llm" and self._llm_config is not None:
            engine = LLMDetectionEngine(
                self._llm_config,
                sensitive_entities=policy.sensitive_entities or None,
                instruction=policy.detector_system_prompt,
                **policy.detection_params,
            )
            return lambda chunk, _policy: engine.detect(chunk)
        return _resolve_engine_tool(policy.detection_engine)

    def detect(
        self, policy: PrivacyPolicy, chunks: list[str]
    ) -> tuple[list[list[PIISpan]], RiskAssessment]:
        """Run the policy's engine over each chunk and assess overall risk."""
        tool = self._resolve_tool(policy)
        spans_per_chunk: list[list[PIISpan]] = []
        for number, chunk in enumerate(chunks, 1):
            logger.debug(
                "Detection (%s): chunk %d/%d",
                policy.detection_engine,
                number,
                len(chunks),
            )
            spans_per_chunk.append(_dedupe_spans(tool(chunk, policy)))
        return spans_per_chunk, _assess_risk(chunks, spans_per_chunk)

    def _node(self, state: PrivacyState) -> PrivacyState:
        """Graph node: write spans and risk into the pipeline state."""
        policy = state.get("policy")
        if policy is None:
            raise ValueError("DetectorAgent requires 'policy' in the state.")
        if state.get("skip_detection") and state.get("spans") is not None:
            logger.debug("Skipping detection during escalation")
            return PrivacyState(skip_detection=False)
        chunks = list(state.get("raw_chunks", []))
        spans_per_chunk, risk = self.detect(policy, chunks)
        query_spans, _ = self.detect(policy, [state.get("query", "")])
        # the query is a single item, and because detection is done in batch
        # we take the result at index 0
        return PrivacyState(
            spans=spans_per_chunk, query_spans=query_spans[0], risk=risk
        )
