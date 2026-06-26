import os
from dataclasses import dataclass

# from getpass import getpass
from typing import ClassVar, Optional, cast

try:
    import torch
except ImportError:
    torch = None

from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_core.language_models.chat_models import BaseChatModel

# HF Models
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_mistralai import ChatMistralAI

# Proprietary Models
from langchain_openai import ChatOpenAI

from ..utils import load_config
from ..ux import loading_model

_OPENAI_MODELS = [
    # GPT-5 series (2026)
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.2",
    "gpt-5",
    "gpt-5-mini",
    # GPT-4 series
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    # Reasoning models
    "o3",
    "o3-mini",
    "o4-mini",
    # Legacy (still supported but being phased out)
    "gpt-3.5-turbo",
]
_ANTHROPIC_MODELS = [
    # Claude 4 (current generation - 2026)
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    # Claude 3.5 (still supported)
    "claude-3-5-sonnet",
    "claude-3-5-haiku",
    # Claude 3 (legacy)
    "claude-3-opus",
    "claude-3-sonnet",
    "claude-3-haiku",
]
_MISTRAL_MODELS = [
    # Current frontier models (2026)
    "mistral-small-4",
    "mistral-large-3",
    "mistral-medium-3.1",
    "mistral-small-3.2",
    # Ministral family
    "ministral-3-14b",
    "ministral-3-8b",
    "ministral-3-3b",
    # Specialist models
    "codestral",
    "codestral-latest",
    # Legacy aliases (still supported)
    "mistral-small-latest",
    "mistral-medium-latest",
    "mistral-large-latest",
]
_COHERE_MODELS = [
    # Command A series (2025-2026)
    "command-a-03-2025",
    "command-a-translate-08-2025",
    "command-a-reasoning-08-2025",
    "command-a-vision-07-2025",
    # Command R series (current)
    "command-r7b-12-2024",
    "command-r-08-2024",
    "command-r-plus-08-2024",
]

loaders = {
    "OPENAI": ChatOpenAI,
    "ANTHROPIC": ChatAnthropic,
    "MISTRAL": ChatMistralAI,
    "COHERE": ChatCohere,
    "HF": ChatHuggingFace,
}


@dataclass
class LLMConfig:
    llm_name: str
    base_url: Optional[str] = None
    provider: Optional[str] = None
    max_new_tokens: Optional[int] = None
    temperature: float = 0.7

    def __post_init__(self):
        self.provider = self.provider or (
            "OPENAI"
            if self.llm_name in _OPENAI_MODELS
            else (
                "ANTHROPIC"
                if self.llm_name in _ANTHROPIC_MODELS
                else (
                    "MISTRAL"
                    if self.llm_name in _MISTRAL_MODELS
                    else (
                        "COHERE"
                        if self.llm_name in _COHERE_MODELS
                        else "HF"
                        if self.base_url is None
                        else None
                    )
                )
            )
        )

        if self.provider is not None:
            self.provider = self.provider.upper()

    @property
    def generation_kwargs(self):
        if self.provider in ["MISTRAL", "ANTHROPIC", "COHERE"]:
            max_token_key = "max_tokens"
        elif self.provider == "HF":
            max_token_key = "max_new_tokens"
        else:
            max_token_key = "max_completion_tokens"
        return {"temperature": self.temperature, max_token_key: self.max_new_tokens}

    @property
    def api_key(self):
        if self.provider:
            LLM._check_key(self.provider)
            return os.environ[f"{self.provider}_API_KEY"]
        else:
            return "EMPTY"


class LLM(BaseChatModel):
    """Class parsing the model name and arguments to load the correct LangChain model"""

    device_count: ClassVar[int] = 0
    nb_devices: ClassVar[Optional[int]] = None

    @classmethod
    def _get_nb_devices(cls) -> int:
        if cls.nb_devices is None:
            if torch is not None and torch.cuda.is_available():
                cls.nb_devices = torch.cuda.device_count()
            else:
                cls.nb_devices = 1
        return cls.nb_devices

    @staticmethod
    def _check_key(provider):
        if f"{provider}_API_KEY" not in os.environ:
            # print(f"Enter your {provider} API key:")
            # os.environ[f"{provider}_API_KEY"] = getpass()
            raise ValueError(
                f"Unable to find the API key for {provider}. Please restart after setting the '{provider}_API_KEY' environment variable."
            )

    @classmethod
    def from_config(
        cls,
        config: str | LLMConfig,
        *,
        hf_return_full_text: bool | None = None,
    ) -> BaseChatModel:
        if isinstance(config, str):
            config = load_config(config, LLMConfig)

        if config.provider == "HF":
            if torch is None:
                raise ImportError(
                    "torch is required for HuggingFace models. "
                    "Install it with: uv pip install 'mmore[cpu]' or uv pip install 'mmore[cu126]'"
                )
            pipeline_kwargs = dict(config.generation_kwargs)
            if hf_return_full_text is not None:
                pipeline_kwargs["return_full_text"] = hf_return_full_text
            with loading_model(f"the answer-generation model ({config.llm_name})"):
                if torch.backends.mps.is_available():
                    return ChatHuggingFace(
                        llm=HuggingFacePipeline.from_model_id(
                            model_id=config.llm_name,
                            task="text-generation",
                            device_map="mps",
                            pipeline_kwargs=pipeline_kwargs,
                        )
                    )
                if torch.cuda.is_available():
                    current_device = cls.device_count
                    cls.device_count = (cls.device_count + 1) % cls._get_nb_devices()
                else:
                    current_device = -1

                return ChatHuggingFace(
                    llm=HuggingFacePipeline.from_model_id(
                        config.llm_name,
                        task="text-generation",
                        device=current_device,
                        pipeline_kwargs=pipeline_kwargs,
                    )
                )
        else:
            loader = loaders.get(cast(str, config.provider), ChatOpenAI)
            return loader(
                model=config.llm_name,
                base_url=config.base_url,
                api_key=config.api_key,
                **config.generation_kwargs,
            )
