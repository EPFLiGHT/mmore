"""Sanitizer.

Pipeline:  analyzer  ->  detector  ->  [sanitizer]  ->  adversarial -> ...
Reads:     policy, raw_chunks, spans
Writes:    sanitized_chunks

One node in the privacy multi-agent pipeline: resolves the policy's
sanitization strategy from the tool registry and applies it to each chunk.
"""

import logging

import dspy
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import Self

from ...rag.llm import LLMConfig
from ..config import PrivacyConfig, as_privacy_config
from ..detection.base import PIISpan
from ..detection.constants import DEFAULT_LLM_CONFIG
from ..sanitization.base import SanitizationTool
from ..sanitization.constants import SANITIZATION_TOOL_NAMES
from ..schemas.policy import PrivacyPolicy
from .base import BaseAgent
from .registry import ToolNotRegisteredError, resolve_tool
from .state import PrivacyState

logger = logging.getLogger(__name__)

_LLM_BACKED_STRATEGY = "synthetic_rewrite"


def _resolve_strategy_tool(strategy: str) -> SanitizationTool:
    """Resolve a strategy short name to its registered sanitization tool."""
    tool_name = SANITIZATION_TOOL_NAMES.get(strategy)
    if tool_name is None:
        raise ToolNotRegisteredError(
            f"Unknown sanitization strategy '{strategy}'. "
            f"Known strategies: {sorted(SANITIZATION_TOOL_NAMES)}"
        )
    return resolve_tool(tool_name)


class SanitizerAgent(BaseAgent):
    """Dispatches to the policy's sanitization strategy via the tool registry."""

    state_schema = PrivacyState
    node_name = "sanitizer"
    fallback_llm_config = DEFAULT_LLM_CONFIG

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
        llm_config = config.sanitization.llm
        if llm_config is None and config.context_analyzer:
            llm_config = config.context_analyzer.llm
        return cls(config, llm_config, checkpointer=checkpointer)

    def sanitize(
        self,
        policy: PrivacyPolicy,
        chunks: list[str],
        spans_per_chunk: list[list[PIISpan]],
    ) -> list[str]:
        """Apply ``policy.sanitization_strategy`` to ``chunks``."""
        tool = _resolve_strategy_tool(policy.sanitization_strategy)
        if policy.sanitization_strategy == _LLM_BACKED_STRATEGY:
            with dspy.context(lm=self.dspy_lm):
                return tool(chunks, spans_per_chunk, policy)
        return tool(chunks, spans_per_chunk, policy)

    def _node(self, state: PrivacyState) -> PrivacyState:
        """Graph node: write sanitized chunks into the pipeline state."""
        policy = state.get("policy")
        if policy is None:
            raise ValueError("SanitizerAgent requires 'policy' in the state.")
        chunks = list(state.get("raw_chunks", []))
        spans_per_chunk = list(state.get("spans") or [[] for _ in chunks])
        if len(spans_per_chunk) != len(chunks):
            raise ValueError(
                f"spans/raw_chunks length mismatch: "
                f"{len(spans_per_chunk)} != {len(chunks)}"
            )
        query_spans = list(state.get("query_spans") or [])
        return PrivacyState(
            sanitized_chunks=self.sanitize(policy, chunks, spans_per_chunk),
            # the query is a single item, and because sanitization is done in batch
            # we take the result at index 0
            sanitized_query=self.sanitize(
                policy, [state.get("query", "")], [query_spans]
            )[0],
        )
