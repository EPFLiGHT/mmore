"""Integration tests for mmore.privacy.agents."""

from typing import Any, Dict, TypedDict
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from mmore.privacy.agents.base import BaseAgent, clear_llm_cache
from mmore.privacy.agents.config import AgentConfig
from mmore.privacy.agents.registry import (
    ToolNotRegisteredError,
    register_tool,
    tool_registry,
)
from mmore.rag.llm import LLMConfig
from mmore.utils import load_config


@pytest.fixture
def isolate_llm_cache():
    clear_llm_cache()
    yield
    clear_llm_cache()


@pytest.fixture
def isolated_tool_registry():
    snapshot = dict(tool_registry)
    tool_registry.clear()
    yield
    tool_registry.clear()
    tool_registry.update(snapshot)


def _cfg(**args: Any) -> AgentConfig:
    base: dict[str, Any] = dict(
        llm=LLMConfig(llm_name="gpt2", max_new_tokens=8, temperature=0.5),
        name="answerer",
        system_prompt="You are helpful.",
    )
    base.update(args)
    return AgentConfig(**base)


def test_system_prompt_is_prepended_to_messages(isolate_llm_cache):
    captured = {}

    class Capturing(FakeListChatModel):
        def _call(self, messages, stop=None, run_manager=None, **kwargs):
            captured["messages"] = messages
            return super()._call(messages, stop, run_manager, **kwargs)

    fake = Capturing(responses=["ok"])
    with patch("mmore.privacy.agents.base._build_chat_model", return_value=fake):
        agent = BaseAgent.from_config(_cfg(system_prompt="SYS"))
        agent.invoke("hello")

    sent = captured["messages"]
    assert isinstance(sent[0], SystemMessage) and sent[0].content == "SYS"
    assert isinstance(sent[1], HumanMessage) and sent[1].content == "hello"


def test_registered_tools_are_bound_to_llm_at_first_invoke(
    isolate_llm_cache, isolated_tool_registry
):
    bound_with = {}

    class Binding(FakeListChatModel):
        def bind_tools(self, tools, **_kwargs):
            bound_with["tools"] = list(tools)
            return self

    @register_tool("greet")
    def greet(name: str) -> str:
        return f"hi {name}"

    fake = Binding(responses=["ok"])
    with patch("mmore.privacy.agents.base._build_chat_model", return_value=fake):
        agent = BaseAgent.from_config(_cfg(tools=["greet"]))
        agent.invoke("trigger")

    assert bound_with["tools"] == [greet]


def test_unknown_tool_in_config_raises_at_from_config(
    isolate_llm_cache, isolated_tool_registry
):
    with pytest.raises(ToolNotRegisteredError):
        BaseAgent.from_config(_cfg(tools=["does_not_exist"]))


def test_agent_config_loads_from_dict_via_dacite():
    raw = {
        "llm": {"llm_name": "gpt2", "max_new_tokens": 32, "temperature": 0.0},
        "name": "sanitizer",
        "system_prompt": "Strip PII.",
        "tools": [],
        "checkpointer": "memory",
    }

    cfg = load_config(raw, AgentConfig)

    assert isinstance(cfg, AgentConfig) and isinstance(cfg.llm, LLMConfig)
    assert cfg.name == "sanitizer"
    assert cfg.checkpointer == "memory"
    assert cfg.llm.temperature == 0.0


def test_memory_checkpointer_persists_state_in_a_thread(isolate_llm_cache):
    fake = FakeListChatModel(responses=["first", "second"])
    cfg = _cfg(checkpointer="memory")
    thread: RunnableConfig = {"configurable": {"thread_id": "t-1"}}

    with patch("mmore.privacy.agents.base._build_chat_model", return_value=fake):
        agent = BaseAgent.from_config(cfg)
        agent.invoke("q1", config=thread)
        agent.invoke("q2", config=thread)

    snapshot = agent.graph.get_state(thread)
    assert [m.content for m in snapshot.values["messages"]] == [
        "q1",
        "first",
        "q2",
        "second",
    ]


def test_sqlite_checkpointer_persists_state_across_agents(tmp_path, isolate_llm_cache):
    db = tmp_path / "check.db"
    fake = FakeListChatModel(responses=["a"])
    cfg = _cfg(checkpointer="sqlite", checkpoint_path=str(db))
    thread: RunnableConfig = {"configurable": {"thread_id": "t-rt"}}

    with patch("mmore.privacy.agents.base._build_chat_model", return_value=fake):
        with BaseAgent.from_config(cfg) as agent_a:
            agent_a.invoke("hello", config=thread)

        with BaseAgent.from_config(cfg) as agent_b:
            snapshot = agent_b.graph.get_state(thread)

    assert [m.content for m in snapshot.values["messages"]] == ["hello", "a"]


def test_lazy_loading_and_dedup_minimize_load_calls(isolate_llm_cache):
    fake = FakeListChatModel(responses=["x"] * 10)
    with patch(
        "mmore.privacy.agents.base._build_chat_model", return_value=fake
    ) as mock:
        agent_x = BaseAgent.from_config(_cfg(name="x"))
        agent_y = BaseAgent.from_config(_cfg(name="y"))
        assert mock.call_count == 0

        agent_x.invoke("q")
        agent_y.invoke("q")
        assert mock.call_count == 1


def test_same_model_different_params_share_one_instance(isolate_llm_cache):
    captured = []

    class Capturing(FakeListChatModel):
        def _call(self, messages, stop=None, run_manager=None, **kwargs):
            captured.append(kwargs)
            return super()._call(messages, stop, run_manager, **kwargs)

    fake = Capturing(responses=["ok"] * 4)
    hot = _cfg(
        name="hot",
        llm=LLMConfig(llm_name="gpt-4o-mini", temperature=0.9, max_new_tokens=128),
    )
    cold = _cfg(
        name="cold",
        llm=LLMConfig(llm_name="gpt-4o-mini", temperature=0.1, max_new_tokens=32),
    )

    with patch(
        "mmore.privacy.agents.base._build_chat_model", return_value=fake
    ) as mock:
        BaseAgent.from_config(hot).invoke("q")
        BaseAgent.from_config(cold).invoke("q")
        assert mock.call_count == 1

    assert captured[0]["temperature"] == 0.9
    assert captured[0]["max_completion_tokens"] == 128
    assert captured[1]["temperature"] == 0.1
    assert captured[1]["max_completion_tokens"] == 32


def test_hf_generation_params_are_bound_as_pipeline_kwargs(isolate_llm_cache):
    captured = []

    class Capturing(FakeListChatModel):
        def _call(self, messages, stop=None, run_manager=None, **kwargs):
            captured.append(kwargs)
            return super()._call(messages, stop, run_manager, **kwargs)

    fake = Capturing(responses=["ok"])
    cfg = _cfg(llm=LLMConfig(llm_name="gpt2", temperature=0.3, max_new_tokens=16))

    with patch("mmore.privacy.agents.base._build_chat_model", return_value=fake):
        BaseAgent.from_config(cfg).invoke("q")

    assert captured[0].get("pipeline_kwargs") == {
        "temperature": 0.3,
        "max_new_tokens": 16,
    }


def test_clear_llm_cache(isolate_llm_cache):
    fake = FakeListChatModel(responses=["x"] * 10)
    with patch(
        "mmore.privacy.agents.base._build_chat_model", return_value=fake
    ) as mock:
        agent = BaseAgent.from_config(_cfg())
        agent.invoke("q")
        assert mock.call_count == 1
        assert agent._llm is not None

        agent.release()
        assert agent._llm is None
        clear_llm_cache()

        new_agent = BaseAgent.from_config(_cfg())
        new_agent.invoke("q")
        assert mock.call_count == 2


# --------------------------------------------------------------------------
# BaseAgent as a generalized pipeline-node base
# --------------------------------------------------------------------------


class _PipeState(TypedDict, total=False):
    value: int
    doubled: int


class _DoublerAgent(BaseAgent):
    """An LLM-less agent on a custom state schema."""

    state_schema = _PipeState
    node_name = "doubler"

    def _node(self, state: _PipeState) -> Dict[str, Any]:
        return {"doubled": state["value"] * 2}


def test_subclass_runs_with_custom_state_schema_and_no_llm():
    agent = _DoublerAgent(config=object(), llm_config=None)

    out = agent.graph.invoke({"value": 21})

    assert out["doubled"] == 42


def test_node_name_class_attr_is_used_as_graph_node_id():
    agent = _DoublerAgent(config=object())

    assert agent.name == "doubler"
    assert "doubler" in agent.graph.get_graph().nodes


def test_default_name_falls_back_to_agent_config_name():
    fake = FakeListChatModel(responses=["ok"])
    with patch("mmore.privacy.agents.base._build_chat_model", return_value=fake):
        agent = BaseAgent.from_config(_cfg(name="answerer"))

    assert agent.name == "answerer"


def test_llm_access_without_config_raises_clear_error():
    agent = _DoublerAgent(config=object(), llm_config=None)

    with pytest.raises(ValueError, match="LLM"):
        _ = agent.llm


def test_node_property_exposes_bound_node_for_composition():
    agent = _DoublerAgent(config=object())

    assert agent.node({"value": 5}) == {"doubled": 10}
