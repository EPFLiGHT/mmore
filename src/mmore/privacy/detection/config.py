"""Configuration dataclass for the PII detection toolkit."""

from dataclasses import dataclass, field
from typing import List, Optional

from ...rag.llm import LLMConfig
from .defaults import DEFAULT_CONFIDENCE_THRESHOLD


@dataclass
class DetectionConfig:
    """Schema for the ``privacy.detection`` block of a YAML config."""

    engine: str
    entity_types: List[str] = field(default_factory=list)
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    llm: Optional[LLMConfig] = None
