"""LLM-backed PII detection engine using DSPy for typed structured output."""

import logging
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import dspy
from pydantic import BaseModel, Field
from typing_extensions import Self

from ...rag.llm import LLMConfig
from ..agents.registry import register_tool
from .base import DetectionEngine, PIISpan
from .config import DetectionConfig

logger = logging.getLogger(__name__)

_DEFAULT_LABELS = [
    "PERSON",
    "PHONE",
    "EMAIL",
    "MRN",
    "DATE",
    "LOCATION",
    "SSN",
    "INSURANCE_ID",
]
_DEFAULT_LLM = LLMConfig(llm_name="Qwen/Qwen2.5-3B-Instruct", max_new_tokens=512)

_pipeline_cache: Dict[str, Any] = {}
_pipeline_cache_lock = threading.Lock()


class _DetectedSpan(BaseModel):
    text: str = Field(description="exact substring of the input that is PII")
    label: str = Field(description="entity type label, e.g. PERSON, EMAIL, MRN")
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="confidence in [0, 1]; not calibrated, but constrained to the range",
    )


def _build_signature() -> Any:
    class DetectPIISignature(dspy.Signature):
        """Find every PII occurrence in the input text. For each, return the
        exact substring (not paraphrased), its entity label, and a confidence."""

        text: str = dspy.InputField(desc="text to scan for PII")
        entity_types: List[str] = dspy.InputField(
            desc="restrict detection to these entity type labels"
        )
        spans: List[_DetectedSpan] = dspy.OutputField(
            desc="list of detected PII spans, each with the exact substring from the input"
        )

    return DetectPIISignature


def _build_demos() -> List[Any]:
    return [
        dspy.Example(
            text="John Doe called from 555-1234 about his MRN 87654321.",
            entity_types=list(_DEFAULT_LABELS),
            spans=[
                _DetectedSpan(text="John Doe", label="PERSON", score=0.95),
                _DetectedSpan(text="555-1234", label="PHONE", score=0.95),
                _DetectedSpan(text="87654321", label="MRN", score=0.95),
            ],
        ).with_inputs("text", "entity_types"),
        dspy.Example(
            text="Patient at 123 Main St emailed jane@example.com on 2024-01-15.",
            entity_types=list(_DEFAULT_LABELS),
            spans=[
                _DetectedSpan(text="123 Main St", label="LOCATION", score=0.9),
                _DetectedSpan(text="jane@example.com", label="EMAIL", score=0.95),
                _DetectedSpan(text="2024-01-15", label="DATE", score=0.9),
            ],
        ).with_inputs("text", "entity_types"),
    ]


def _build_predictor() -> Any:
    predictor = dspy.Predict(_build_signature())
    predictor.demos = _build_demos()
    return predictor


def _load_local_hf_pipeline(model_name: str) -> Any:
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


def _get_or_load_pipeline(model_name: str) -> Any:
    cached = _pipeline_cache.get(model_name)
    if cached is not None:
        return cached
    with _pipeline_cache_lock:
        cached = _pipeline_cache.get(model_name)
        if cached is None:
            cached = _load_local_hf_pipeline(model_name)
            _pipeline_cache[model_name] = cached
        return cached


def clear_llm_engine_cache() -> None:
    """Drop all cached transformers pipelines."""
    with _pipeline_cache_lock:
        _pipeline_cache.clear()


class _LocalHFLM(dspy.BaseLM):
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
    def pipe(self) -> Any:
        return _get_or_load_pipeline(self._model_name)

    def forward(self, prompt=None, messages=None, **kwargs):
        merged = {**self.kwargs, **kwargs}
        max_new_tokens = int(merged.get("max_tokens", 512) or 512)
        temperature = float(merged.get("temperature", 0.0) or 0.0)
        do_sample = temperature > 0.0

        chat_input = messages or [{"role": "user", "content": prompt or ""}]
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "return_full_text": False,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        raw = self.pipe(chat_input, **gen_kwargs)
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


def _build_dspy_lm(llm_config: LLMConfig) -> Any:
    """Build a DSPy ``BaseLM`` from an ``LLMConfig``.

    ``provider="HF"`` (no ``base_url``) routes to ``_LocalHFLM`` for local
    transformers; everything else goes through ``dspy.LM`` (litellm).
    """
    if llm_config.provider == "HF" and llm_config.base_url is None:
        return _LocalHFLM(
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


class LLMDetectionEngine(DetectionEngine):
    """Detect PII spans by prompting an LLM with a typed DSPy signature.

    Each instance carries its own ``LLMConfig``, ``entity_types`` and
    ``confidence_threshold``; pipelines with the same ``model_name`` are
    shared via ``_pipeline_cache``.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        entity_types: Optional[Sequence[str]] = None,
        confidence_threshold: float = 0.7,
    ):
        self._llm_config = llm_config
        self._entity_types: List[str] = (
            list(entity_types) if entity_types else list(_DEFAULT_LABELS)
        )
        self._confidence_threshold = confidence_threshold

    @classmethod
    def from_config(cls, config: DetectionConfig) -> Self:
        """Build an engine from a ``DetectionConfig``."""
        if config.llm is None:
            raise ValueError("DetectionConfig.llm must be set when engine='llm'")
        return cls(
            llm_config=config.llm,
            entity_types=config.entity_types or None,
            confidence_threshold=config.confidence_threshold,
        )

    def detect(self, text: str) -> List[PIISpan]:
        lm = _build_dspy_lm(self._llm_config)
        predictor = _build_predictor()
        try:
            with dspy.context(lm=lm):
                prediction = predictor(text=text, entity_types=self._entity_types)
        except Exception as e:
            logger.warning("LLM detection failed (%s); returning no spans", e)
            return []

        spans: List[PIISpan] = []
        for s in getattr(prediction, "spans", None) or []:
            try:
                fragment = str(s.text)
                label = str(s.label)
                score = float(s.score)
            except (AttributeError, TypeError, ValueError):
                continue
            if not fragment:
                continue
            score = max(0.0, min(1.0, score))
            if score < self._confidence_threshold:
                continue
            start = text.find(fragment)
            if start < 0:
                logger.debug(
                    "LLM emitted fragment %r not found in source text", fragment
                )
                continue
            spans.append(
                PIISpan(
                    start=start, end=start + len(fragment), label=label, score=score
                )
            )
        return spans


@register_tool("detect_pii_llm")
def detect_pii_llm(text: str) -> List[PIISpan]:
    """Detect PII spans in ``text`` using a default-configured LLM engine.

    Agents needing per-config behavior should be wired by setup code that
    builds an ``LLMDetectionEngine.from_config(detection_cfg)`` and registers
    its ``detect()`` function under a distinct tool name, e.g.::

        engine = LLMDetectionEngine.from_config(detection_cfg)
        register_tool("detect_pii_llm_custom", engine.detect)
    """
    return LLMDetectionEngine(_DEFAULT_LLM).detect(text)
