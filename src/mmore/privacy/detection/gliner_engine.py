"""GLiNER-based PII detection engine."""

import logging
import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from typing_extensions import Self

from ..agents.registry import register_tool
from .base import DetectionEngine, PIISpan
from .config import DetectionConfig
from .defaults import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_GLINER_MODEL,
    DEFAULT_LABELS,
)

if TYPE_CHECKING:
    from gliner.model import BaseEncoderGLiNER

logger = logging.getLogger(__name__)

_model_cache: Dict[str, "BaseEncoderGLiNER"] = {}
_model_cache_lock = threading.Lock()


def _load_gliner_model(model_name: str) -> "BaseEncoderGLiNER":
    import torch
    from gliner import GLiNER

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return GLiNER.from_pretrained(model_name).to(device)


def _get_or_load_model(model_name: str) -> "BaseEncoderGLiNER":
    cached = _model_cache.get(model_name)
    if cached is not None:
        return cached
    with _model_cache_lock:
        cached = _model_cache.get(model_name)
        if cached is None:
            cached = _load_gliner_model(model_name)
            _model_cache[model_name] = cached
        return cached


def clear_gliner_cache() -> None:
    """Drop all cached GLiNER models."""
    with _model_cache_lock:
        _model_cache.clear()


class GLiNEREngine(DetectionEngine):
    """Detect PII spans with a GLiNER model.

    Each instance carries its own ``entity_types`` and ``confidence_threshold``,
    models with the same ``model_name`` are shared via ``_models_cache``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GLINER_MODEL,
        entity_types: Optional[Sequence[str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self._model_name = model_name
        self._entity_types: List[str] = (
            list(entity_types) if entity_types else list(DEFAULT_LABELS)
        )
        self._confidence_threshold = confidence_threshold

    @classmethod
    def from_config(cls, config: DetectionConfig) -> Self:
        """Build an engine from a ``DetectionConfig``."""
        return cls(
            entity_types=config.entity_types or None,
            confidence_threshold=config.confidence_threshold,
        )

    @property
    def model(self) -> "BaseEncoderGLiNER":
        """Lazy-load and cache the LLM on first access."""
        return _get_or_load_model(self._model_name)

    def detect(self, text: str) -> List[PIISpan]:
        raw = self.model.predict_entities(
            text=text,
            labels=self._entity_types,
            threshold=self._confidence_threshold,
            multi_label=False,
        )
        return [
            PIISpan(
                start=int(r["start"]),
                end=int(r["end"]),
                label=str(r["label"]),
                score=float(r["score"]),
            )
            for r in raw
        ]


@register_tool("detect_pii_gliner")
def detect_pii_gliner(text: str) -> List[PIISpan]:
    """Detect PII spans in ``text`` using a default-configured GLiNER engine.

    Agents needing per-config behavior should be wired by setup code that
    builds a ``GLiNEREngine.from_config(detection_cfg)`` and registers its
    ``detect()`` function under a distinct tool name, e.g.::

        engine = GLiNEREngine.from_config(detection_cfg)
        register_tool("detect_pii_gliner_custom", engine.detect)
    """
    return GLiNEREngine().detect(text)
