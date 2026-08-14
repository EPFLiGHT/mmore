"""Per-agent configuration dataclass."""

from dataclasses import dataclass, field

from ...rag.llm import LLMConfig


@dataclass
class AgentConfig:
    """General definition of an agent in the privacy system.

    This config will most likely not be directly used in the privacy pipeline
    as we have a ``PrivacyConfig``. However it serves as a template in case we
    want to leverage the Agent integration for other purposes in the future."""

    llm: LLMConfig
    name: str = "agent"
    system_prompt: str = ""
    tools: list[str] = field(default_factory=list)
    checkpointer: str | None = None
    checkpoint_path: str | None = None
