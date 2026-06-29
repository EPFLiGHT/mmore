"""Sanitizer.

Pipeline:  analyzer  ->  detector  ->  [sanitizer]  ->  ... TODO: complete once implemented
Reads:     policy, raw_chunks, spans
Writes:    sanitized_chunks

One node in the privacy multi-agent pipeline: resolves the policy's
sanitization strategy from the tool registry and applies it to each chunk.
"""

import logging
from typing import Callable, List, Optional, Union

import dspy
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import Self

from ...rag.llm import LLMConfig
from ...utils import load_config
from ..config import PrivacyConfig
from ..detection.base import PIISpan
from ..dspy_llm import build_dspy_lm
from ..policy import PrivacyPolicy
from ..sanitization.constants import SANITIZATION_TOOL_NAMES
from .base import BaseAgent
from .registry import tool_registry
from .state import PrivacyState

logger = logging.getLogger(__name__)


# ========================================================================
# Helpers
# ========================================================================


def _resolve_strategy_tool(strategy_short: str) -> Callable[..., List[str]]:
    """Resolve the strategy short name to a registered sanitization tool callable."""
    tool_name = SANITIZATION_TOOL_NAMES.get(strategy_short)
    if tool_name is None:
        raise KeyError(
            f"Unknown sanitization strategy '{strategy_short}'. "
            f"Known strategies: {sorted(SANITIZATION_TOOL_NAMES)}"
        )
    if tool_name not in tool_registry:
        raise KeyError(
            f"Sanitization tool '{tool_name}' is not registered. "
            f"Available tools: {sorted(tool_registry.keys())}"
        )
    return tool_registry[tool_name]


# ========================================================================
# Agent
# ========================================================================


class SanitizerAgent(BaseAgent):
    """Dispatches to the policy's sanitization strategy via the tool registry."""

    state_schema = PrivacyState
    node_name = "sanitizer"

    def __init__(
        self,
        config: PrivacyConfig,
        llm_config: Optional[LLMConfig] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self._dspy_lm: Optional[dspy.BaseLM] = None
        super().__init__(config, llm_config=llm_config, checkpointer=checkpointer)

    @classmethod
    def from_config(
        cls,
        config: Union[PrivacyConfig, str, dict],
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> Self:
        if not isinstance(config, PrivacyConfig):
            config = load_config(config, PrivacyConfig)
        llm_config = config.sanitization.llm if config.sanitization else None
        return cls(config, llm_config, checkpointer=checkpointer)

    def _ensure_dspy_lm(self) -> dspy.BaseLM:
        """Lazily build the DSPy LM when sanitization method requires an LLM."""
        if self._dspy_lm is None:
            if self._llm_config is None:
                raise ValueError("Sanitizer strategy requires an LLM.")
            self._dspy_lm = build_dspy_lm(self._llm_config)
        return self._dspy_lm

    def sanitize(
        self,
        policy: PrivacyPolicy,
        chunks: List[str],
        spans_per_chunk: List[List[PIISpan]],
    ) -> List[str]:
        """Apply ``policy.sanitization_strategy`` to ``chunks``."""
        tool = _resolve_strategy_tool(policy.sanitization_strategy)
        if policy.sanitization_strategy == "synthetic_rewrite":
            with dspy.context(lm=self._ensure_dspy_lm()):
                return tool(chunks, spans_per_chunk, policy)
        return tool(chunks, spans_per_chunk, policy)

    def _node(self, state: PrivacyState) -> PrivacyState:
        """Graph node: write sanitized chunks into the pipeline state."""
        policy = state.get("policy")
        if policy is None:
            raise ValueError("SanitizerAgent requires 'policy' in the state.")
        chunks = list(state.get("raw_chunks", []))
        spans_per_chunk = list(state.get("spans") or [[] for _ in chunks])
        sanitized = self.sanitize(policy, chunks, spans_per_chunk)
        return PrivacyState(sanitized_chunks=sanitized)
