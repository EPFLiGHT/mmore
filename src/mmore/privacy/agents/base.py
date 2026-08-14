"""Base class for privacy agents.

A ``BaseAgent`` is one LangGraph node. By default it calls an LLM on the
message history. Subclasses override ``state_schema`` and ``_node`` to act on
a different state (with or without an LLM), and ``node`` exposes the bound
node so several agents can be combined into one pipeline graph.
"""

import logging
from typing import Annotated, Callable, ClassVar, TypedDict

import dspy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Self

from ...rag.llm import LLM, LLMConfig
from ...utils import load_config
from ..dspy_llm import build_dspy_lm, get_local_hf_pipeline
from ..model_cache import MODEL_REGISTRY
from .checkpointer import build_checkpointer
from .config import AgentConfig
from .registry import resolve_tools

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "agent_llm"


def _llm_cache_key(config: LLMConfig) -> str:
    return f"{_CACHE_PREFIX}:{config.llm_name}:{config.base_url}:{config.provider}"


def _build_chat_model(config: LLMConfig) -> BaseChatModel:
    """Build the chat model for ``config``.

    Local HF models wrap the shared registry pipeline, so the weights are
    loaded once and reused by every agent and engine using the same model.
    """
    if config.provider == "HF" and config.base_url is None:
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        pipe = get_local_hf_pipeline(config.llm_name)
        return ChatHuggingFace(
            llm=HuggingFacePipeline(pipeline=pipe, model_id=config.llm_name),
            tokenizer=pipe.tokenizer,
        )
    return LLM.from_config(config)


def clear_llm_cache() -> None:
    """Drop all cached agent chat models."""
    MODEL_REGISTRY.clear(prefix=_CACHE_PREFIX)


class AgentState(TypedDict):
    """Default typed state shared by all single-node privacy agents."""

    messages: Annotated[list[BaseMessage], add_messages]


class NodeOutput(TypedDict, total=False):
    """Generic partial state update returned by any agent node."""

    messages: Annotated[list[BaseMessage], add_messages]


class BaseAgent:
    """Single LangGraph node compiled from a config."""

    state_schema: ClassVar[type] = AgentState
    node_name: ClassVar[str | None] = None
    # Used when the agent needs a DSPy LM but none was configured, None
    # makes a missing LLM an error instead (i.e. analyzer and adversary leave it to None).
    fallback_llm_config: ClassVar[LLMConfig | None] = None

    def __init__(
        self,
        config,
        llm_config: LLMConfig | None = None,
        tools: list[Callable] | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self.config = config
        self._llm_config = llm_config
        self._tools: list[Callable] = list(tools) if tools else []
        self._llm: BaseChatModel | None = None
        self._dspy_lm: dspy.BaseLM | None = None
        self.checkpointer = checkpointer
        self._owns_checkpointer = False
        self.graph = self._build_graph()

    @property
    def name(self) -> str:
        return (
            self.node_name or getattr(self.config, "name", None) or type(self).__name__
        )

    @property
    def system_prompt(self) -> str:
        return getattr(self.config, "system_prompt", "") or ""

    @classmethod
    def from_config(
        cls,
        config,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> Self:
        if not isinstance(config, AgentConfig):
            config = load_config(config, AgentConfig)

        owns_checkpointer = False
        if checkpointer is None and config.checkpointer is not None:
            checkpointer = build_checkpointer(config)
            owns_checkpointer = True

        agent = cls(
            config,
            config.llm,
            resolve_tools(config.tools) if config.tools else [],
            checkpointer,
        )
        agent._owns_checkpointer = owns_checkpointer
        return agent

    @property
    def llm_config(self) -> LLMConfig:
        """The agent's LLM config.

        Raises:
            ValueError: if the agent has no LLM configured. An agent whose
                node never touches an LLM (e.g. the Detector) is valid.
        """
        if self._llm_config is None:
            raise ValueError(f"{type(self).__name__} has no LLM configured.")
        return self._llm_config

    @property
    def llm(self) -> BaseChatModel:
        """Lazy-load and cache the chat model on first access."""
        if self._llm is None:
            config = self.llm_config
            self._llm = MODEL_REGISTRY.get_or_load(
                _llm_cache_key(config), lambda: _build_chat_model(config)
            )
        return self._llm

    @property
    def dspy_lm(self) -> dspy.BaseLM:
        """Lazy-build and cache the DSPy LM the agent's predictors run under."""
        if self._dspy_lm is None:
            if self._llm_config is None:
                if self.fallback_llm_config is None:
                    raise ValueError(
                        f"{type(self).__name__} requires an LLM but none is configured."
                    )
                logger.warning(
                    "No LLM configured for the %s, falling back to %r",
                    self.name,
                    self.fallback_llm_config.llm_name,
                )
                self._llm_config = self.fallback_llm_config
            self._dspy_lm = build_dspy_lm(self._llm_config)
        return self._dspy_lm

    def release(self) -> None:
        """Release LLM and close checkpointer resources if necessary."""
        if self._owns_checkpointer and self.checkpointer is not None:
            conn = getattr(self.checkpointer, "conn", None)
            if conn is not None:
                conn.close()
        self._llm = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.release()

    @property
    def node(self) -> Callable[..., NodeOutput]:
        """The bound node callable, for composing into a larger graph."""
        return self._node

    def _build_graph(self):
        graph = StateGraph(self.state_schema)
        graph.add_node(self.name, self._node)
        graph.add_edge(START, self.name)
        graph.add_edge(self.name, END)
        return graph.compile(checkpointer=self.checkpointer)

    def _node(self, state) -> NodeOutput:
        messages: list[BaseMessage] = list(state["messages"])
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt), *messages]
        llm = self.llm.bind_tools(self._tools) if self._tools else self.llm
        if self._llm_config is not None:
            llm = llm.bind(**self._llm_config.bind_kwargs)
        return NodeOutput(messages=[llm.invoke(messages)])

    def invoke(
        self,
        query: str | AgentState,
        config: RunnableConfig | None = None,
    ) -> dict:
        """Run the agent graph on a user message or a pre-built state dict."""
        if isinstance(query, str):
            query = AgentState(messages=[HumanMessage(content=query)])
        return self.graph.invoke(query, config=config)
