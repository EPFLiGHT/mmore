"""Judge LLM factory — keeps HF judge settings out of the main RAG LLM path."""

from langchain_core.language_models.chat_models import BaseChatModel

from ...utils import load_config
from ..llm import LLM, LLMConfig


def judge_llm_from_config(config: str | LLMConfig) -> BaseChatModel:
    """Load the judge LLM (``judge.llm`` in config).

    Same as ``LLM.from_config`` except HuggingFace models use
    ``return_full_text=False`` so generation-only output is easier to parse as JSON.
    Non-HF providers are unchanged.
    """
    if isinstance(config, str):
        config = load_config(config, LLMConfig)
    if config.provider != "HF":
        return LLM.from_config(config)
    return LLM.from_config(config, hf_return_full_text=False)
