"""HuggingFace ``openai/privacy-filter`` PII detection engine."""

import logging
import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from typing_extensions import Self

from ..agents.registry import register_tool
from .base import DetectionEngine, PIISpan
from .config import DetectionConfig
from .defaults import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_OPENAI_FILTER_MODEL

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import TokenClassificationPipeline


def _load_openai_filter_pipeline(model_name: str) -> TokenClassificationPipeline:
    from transformers import pipeline

    return pipeline(
        task="token-classification",
        model=model_name,
    )


_pipeline_cache: Dict[str, TokenClassificationPipeline] = {}
_pipeline_cache_lock = threading.Lock()


def _get_or_load_pipeline(model_name: str) -> TokenClassificationPipeline:
    cached = _pipeline_cache.get(model_name)
    if cached is not None:
        return cached
    else:
        with _pipeline_cache_lock:
            cached = _pipeline_cache.get(model_name)
            if cached is None:
                cached = _load_openai_filter_pipeline(model_name)
                _pipeline_cache[model_name] = cached
            return cached


def clear_openai_filter_cache() -> None:
    """Drop all cached HF pipelines."""
    with _pipeline_cache_lock:
        _pipeline_cache.clear()


class OpenAIFilterEngine(DetectionEngine):
    """Detect PII spans with the token classification model
    ``openai/privacy-filter`` from HuggingFace.

    Each instance carries its own ``entity_types`` and ``confidence_threshold``,
    pipelines with the same ``model_name`` are shared via ``_pipeline_cache``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_FILTER_MODEL,
        entity_types: Optional[Sequence[str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        if entity_types:
            logger.warning(
                "OpenAIFilterEngine ignores entity_types=%r; "
                "the model emits its own label taxonomy.",
                list(entity_types),
            )
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold

    @classmethod
    def from_config(cls, config: DetectionConfig) -> Self:
        return cls(
            entity_types=config.entity_types or None,
            confidence_threshold=config.confidence_threshold,
        )

    @property
    def pipeline(self) -> TokenClassificationPipeline:
        return _get_or_load_pipeline(self._model_name)

    def detect(self, text: str) -> List[PIISpan]:
        raw = self.pipeline(text)
        spans: List[PIISpan] = []
        for r in raw:
            score = float(r["score"])
            if score < self._confidence_threshold:
                continue
            label = str(r.get("entity_group") or r.get("entity") or "")
            spans.append(
                PIISpan(
                    start=int(r["start"]),
                    end=int(r["end"]),
                    label=label,
                    score=score,
                )
            )
        return spans


@register_tool("detect_pii_openai_filter")
def detect_pii_openai_filter(text: str) -> List[PIISpan]:
    """Detect PII spans in ``text`` using a default-configured openai/privacy-filter engine.

    Agents needing per-config behavior should be wired by setup code that
    builds an ``OpenAIFilterEngine.from_config(detection_cfg)`` and registers
    its ``detect()`` function under a distinct tool name, e.g.::

        engine = OpenAIFilterEngine.from_config(detection_cfg)
        register_tool("detect_pii_openai_filter_custom", engine.detect)
    """
    return OpenAIFilterEngine().detect(text)
