"""HuggingFace ``openai/privacy-filter`` PII detection engine."""

import logging
from typing import TYPE_CHECKING

from typing_extensions import Self

from ..agents.registry import register_tool
from ..config import DetectionConfig, DetectionEngineType
from ..model_cache import MODEL_REGISTRY
from ..schemas.policy import PrivacyPolicy
from .base import DetectionEngine, PIISpan
from .constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_OPENAI_FILTER_MODEL,
    threshold_or_default,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import TokenClassificationPipeline

_CACHE_PREFIX = DetectionEngineType.OPENAI_FILTER.value


def _load_openai_filter_pipeline(model_name: str) -> "TokenClassificationPipeline":
    from transformers import pipeline

    return pipeline(task="token-classification", model=model_name)


def clear_openai_filter_cache() -> None:
    """Drop all cached HF pipelines."""
    MODEL_REGISTRY.clear(prefix=_CACHE_PREFIX)


class OpenAIFilterEngine(DetectionEngine):
    """Detect PII spans with the token classification model
    ``openai/privacy-filter`` from HuggingFace.

    The model has a fixed label set so entity selection is not configurable.
    Each instance carries its own ``confidence_threshold``, pipelines with the
    same ``model_name`` are shared via ``MODEL_REGISTRY``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_FILTER_MODEL,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold

    @classmethod
    def from_config(cls, config: DetectionConfig) -> Self:
        return cls(
            confidence_threshold=threshold_or_default(config.confidence_threshold)
        )

    @property
    def pipeline(self) -> "TokenClassificationPipeline":
        return MODEL_REGISTRY.get_or_load(
            f"{_CACHE_PREFIX}:{self._model_name}",
            lambda: _load_openai_filter_pipeline(self._model_name),
        )

    def detect(self, text: str) -> list[PIISpan]:
        spans: list[PIISpan] = []
        for prediction in self.pipeline(text):
            score = float(prediction["score"])
            if score < self._confidence_threshold:
                continue
            spans.append(
                PIISpan(
                    start=int(prediction["start"]),
                    end=int(prediction["end"]),
                    label=str(
                        prediction.get("entity_group") or prediction.get("entity") or ""
                    ),
                    score=score,
                )
            )
        return spans


@register_tool("detect_pii_openai_filter")
def detect_pii_openai_filter(text: str, policy: PrivacyPolicy) -> list[PIISpan]:
    """Detect PII spans in ``text`` using an openai/privacy-filter engine
    configured from ``policy``."""
    if policy.sensitive_entities:
        logger.debug("OpenAI privacy-filter has a fixed sensitive label list.")
    engine = OpenAIFilterEngine(**policy.detection_params)
    return engine.detect(text)
