"""Build a DSPy ``BaseLM`` from an ``LLMConfig``."""

import logging
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict

import dspy

from ..rag.llm import LLMConfig

if TYPE_CHECKING:
    from transformers import TextGenerationPipeline

logger = logging.getLogger(__name__)

_pipeline_cache: Dict[str, TextGenerationPipeline] = {}
_pipeline_cache_lock = threading.Lock()


def _load_local_hf_pipeline(model_name: str) -> TextGenerationPipeline:
    import torch
    from transformers import pipeline

    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.bfloat16
    elif torch.cuda.is_available():
        device, dtype = 0, torch.bfloat16
    else:
        device, dtype = -1, torch.float32
    return pipeline(
        task="text-generation",
        model=model_name,
        device=device,
        torch_dtype=dtype,
    )


def _get_or_load_pipeline(model_name: str) -> TextGenerationPipeline:
    cached = _pipeline_cache.get(model_name)
    if cached is not None:
        return cached
    with _pipeline_cache_lock:
        cached = _pipeline_cache.get(model_name)
        if cached is None:
            cached = _load_local_hf_pipeline(model_name)
            _pipeline_cache[model_name] = cached
        return cached


def clear_dspy_lm_cache() -> None:
    """Drop all cached transformers pipelines."""
    with _pipeline_cache_lock:
        _pipeline_cache.clear()


class LocalHFLM(dspy.BaseLM):
    """``dspy.BaseLM`` that runs a local transformers chat model."""

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):
        super().__init__(
            model=f"local-hf/{model_name}",
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=False,
        )
        self._model_name = model_name

    @property
    def supports_response_schema(self) -> bool:
        return False

    @property
    def supports_function_calling(self) -> bool:
        return False

    @property
    def pipe(self) -> TextGenerationPipeline:
        return _get_or_load_pipeline(self._model_name)

    def forward(self, prompt=None, messages=None, **kwargs):
        merged = {**self.kwargs, **kwargs}
        max_new_tokens = int(merged.get("max_tokens", 512))
        temperature = float(merged.get("temperature", 0.0))
        do_sample = temperature > 0.0

        chat_input = messages or [{"role": "user", "content": prompt or ""}]
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "return_full_text": False,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        raw: list = self.pipe(chat_input, **gen_kwargs)
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        first = raw[0] if raw else {}
        text = first.get("generated_text", "")
        if isinstance(text, list):
            assistant_turns = [m for m in text if m.get("role") == "assistant"]
            text = assistant_turns[-1].get("content", "") if assistant_turns else ""

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text, role="assistant"),
                    finish_reason="stop",
                )
            ],
            model=self.model,
            usage={},
            _hidden_params={},
        )


def build_dspy_lm(llm_config: LLMConfig) -> dspy.BaseLM:
    """Build a DSPy ``BaseLM`` from an ``LLMConfig``."""
    if llm_config.provider == "HF" and llm_config.base_url is None:
        return LocalHFLM(
            model_name=llm_config.llm_name,
            max_tokens=llm_config.max_new_tokens or 512,
            temperature=llm_config.temperature,
        )

    provider_to_prefix = {
        "OPENAI": "openai",
        "ANTHROPIC": "anthropic",
        "MISTRAL": "mistral",
        "COHERE": "cohere",
    }
    prefix = provider_to_prefix.get(llm_config.provider or "")
    model = f"{prefix}/{llm_config.llm_name}" if prefix else llm_config.llm_name

    kwargs: dict = {
        "model": model,
        "temperature": llm_config.temperature,
    }
    if llm_config.max_new_tokens is not None:
        kwargs["max_tokens"] = llm_config.max_new_tokens
    if llm_config.base_url is not None:
        kwargs["api_base"] = llm_config.base_url
    if llm_config.provider:
        kwargs["api_key"] = llm_config.api_key

    return dspy.LM(**kwargs)
