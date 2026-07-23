"""GLiNER-based PII detection engine."""

import logging
from typing import TYPE_CHECKING, Sequence

from typing_extensions import Self

from ...ux import loading_model
from ..agents.registry import register_tool
from ..config import DetectionConfig, DetectionEngineType
from ..model_cache import MODEL_REGISTRY
from ..schemas.policy import PrivacyPolicy
from .base import DetectionEngine, PIISpan
from .constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_ENTITIES,
    DEFAULT_GLINER_MODEL,
    threshold_or_default,
)

if TYPE_CHECKING:
    from gliner.model import BaseEncoderGLiNER

logger = logging.getLogger(__name__)

_CACHE_PREFIX = DetectionEngineType.GLINER.value


def _load_gliner_model(model_name: str) -> "BaseEncoderGLiNER":
    import torch
    from gliner import GLiNER

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    with loading_model(f"the PII detection model ({model_name})"):
        return GLiNER.from_pretrained(model_name).to(device)


def clear_gliner_cache() -> None:
    """Drop all cached GLiNER models."""
    MODEL_REGISTRY.clear(prefix=_CACHE_PREFIX)


class GLiNEREngine(DetectionEngine):
    """Detect PII spans with a GLiNER model.

    Each instance carries its own ``entity_types`` and ``confidence_threshold``.
    The model is shared across instances via ``MODEL_REGISTRY``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GLINER_MODEL,
        sensitive_entities: Sequence[str] | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        multi_label: bool = False,
    ):
        self._model_name = model_name
        self._sensitive_entities = list(sensitive_entities or DEFAULT_ENTITIES)
        self._confidence_threshold = confidence_threshold
        self._multi_label = multi_label

    @classmethod
    def from_config(cls, config: DetectionConfig) -> Self:
        """Build an engine from a ``DetectionConfig``."""
        return cls(
            sensitive_entities=config.entity_types or None,
            confidence_threshold=threshold_or_default(config.confidence_threshold),
        )

    @property
    def model(self) -> "BaseEncoderGLiNER":
        """Lazy-load and cache the model on first access."""
        return MODEL_REGISTRY.get_or_load(
            f"{_CACHE_PREFIX}:{self._model_name}",
            lambda: _load_gliner_model(self._model_name),
        )

    def detect(self, text: str) -> list[PIISpan]:
        predictions = self.model.predict_entities(
            text=text,
            labels=self._sensitive_entities,
            threshold=self._confidence_threshold,
            multi_label=self._multi_label,
        )
        return [
            PIISpan(
                start=int(p["start"]),
                end=int(p["end"]),
                label=str(p["label"]),
                score=float(p["score"]),
            )
            for p in predictions
        ]


@register_tool("detect_pii_gliner")
def detect_pii_gliner(text: str, policy: PrivacyPolicy) -> list[PIISpan]:
    """Detect PII spans in ``text`` using a GLiNER engine configured from ``policy``."""
    engine = GLiNEREngine(
        sensitive_entities=policy.sensitive_entities or None,
        **policy.detection_params,
    )
    return engine.detect(text)
