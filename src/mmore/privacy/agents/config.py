"""Per-agent configuration dataclass."""

from dataclasses import dataclass, field
from typing import List, Optional

from ...rag.llm import LLMConfig


@dataclass
class AgentConfig:
    """General definition of a agent in the privacy system.

    This config will most likely not be used in the privacy pipeline as we have
    a ``PrivacyConfig``. However it serves as a template in case we want to
    leverage the Agent integration for other purposes."""

    llm: LLMConfig
    name: str = "agent"
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)
    checkpointer: Optional[str] = None
    checkpoint_path: Optional[str] = None
