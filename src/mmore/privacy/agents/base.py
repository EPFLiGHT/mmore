"""Base class for privacy agents.

A ``BaseAgent`` is one LangGraph node. By default it calls an LLM on the
message history. Subclasses override ``state_schema`` and ``_node`` to act on
a different state (with or without an LLM), and ``node`` exposes the bound
node so several agents can be combined into one pipeline graph.
"""

from typing import (
    Annotated,
    Callable,
    ClassVar,
    List,
    Optional,
    TypedDict,
    Union,
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Self

from ...rag.llm import LLM, LLMConfig
from ...utils import load_config
from .._cache import MODEL_REGISTRY
from ..dspy_llm import get_local_hf_pipeline
from .checkpointer import build_checkpointer
from .config import AgentConfig
from .registry import resolve_tools

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


def _get_or_load_llm(config: LLMConfig) -> BaseChatModel:
    return MODEL_REGISTRY.get_or_load(
        _llm_cache_key(config), lambda: _build_chat_model(config)
    )


def clear_llm_cache() -> None:
    """Drop all cached agent chat models."""
    MODEL_REGISTRY.clear(prefix=_CACHE_PREFIX)


class AgentState(TypedDict):
    """Default typed state shared by all single-node privacy agents."""

    messages: Annotated[List[BaseMessage], add_messages]


class NodeOutput(TypedDict, total=False):
    """Generic partial state update returned by any agent node."""

    messages: Annotated[List[BaseMessage], add_messages]


class BaseAgent:
    """Single LangGraph node compiled from a config."""

    state_schema: ClassVar[type] = AgentState
    node_name: ClassVar[Optional[str]] = None

    def __init__(
        self,
        config,
        llm_config: Optional[LLMConfig] = None,
        tools: Optional[List[Callable]] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self.config = config
        self._llm_config = llm_config
        self._tools: List[Callable] = list(tools) if tools else []
        self._llm: Optional[BaseChatModel] = None
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
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> Self:
        if not isinstance(config, AgentConfig):
            config = load_config(config, AgentConfig)

        owns_checkpointer = False
        if checkpointer is None and config.checkpointer is not None:
            checkpointer = build_checkpointer(config)
            owns_checkpointer = True

        tools = resolve_tools(config.tools) if config.tools else []

        agent = cls(config, config.llm, tools, checkpointer)
        agent._owns_checkpointer = owns_checkpointer
        return agent

    @property
    def llm(self) -> BaseChatModel:
        """Lazy-load and cache the LLM on first access.

        Raises:
            ValueError: if the agent has no LLM configured. An agent whose
                node never touches the LLM (e.g. the Detector) is valid.
        """
        if self._llm is None:
            if self._llm_config is None:
                raise ValueError(f"{type(self).__name__} has no LLM configured ")
            self._llm = _get_or_load_llm(self._llm_config)
        return self._llm

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
        messages: List[BaseMessage] = list(state["messages"])
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt), *messages]
        llm = self.llm.bind_tools(self._tools) if self._tools else self.llm
        llm = llm.bind(**self._llm_config.bind_kwargs)
        response = llm.invoke(messages)
        return NodeOutput(messages=[response])

    def invoke(
        self,
        query: Union[str, AgentState],
        config: Optional[RunnableConfig] = None,
    ) -> dict:
        """Run the agent graph on the given query.

        Args:
            query: A user message string or a pre-built state dict.
            config: Optional LangGraph runtime config.

        Returns:
            The final graph state dict.
        """
        if isinstance(query, str):
            input_state: AgentState = {"messages": [HumanMessage(content=query)]}
        else:
            input_state = query
        return self.graph.invoke(input_state, config=config)
