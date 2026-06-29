"""Top-level configuration for the privacy pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from ..rag.llm import LLMConfig


class DetectionEngineType(str, Enum):
    """The supported PII detection engines."""

    GLINER = "gliner"
    LLM = "llm"
    OPENAI_FILTER = "openai_filter"
    PRESIDIO = "presidio"


class SanitizationStrategyType(str, Enum):
    """The supported sanitization strategies."""

    TOKEN_MASKING = "token_masking"
    ENTITY_REPLACEMENT = "entity_replacement"
    SYNTHETIC_REWRITE = "synthetic_rewrite"
    PRESIDIO = "presidio"


@dataclass
class AnalyzerConfig:
    llm: LLMConfig
    system_prompt: Optional[str] = None


@dataclass
class DetectionConfig:
    engine: Optional[DetectionEngineType] = None
    confidence_threshold: Optional[float] = None
    entity_types: List[str] = field(default_factory=list)
    llm: Optional[LLMConfig] = None


@dataclass
class SanitizationConfig:
    strategy: Optional[SanitizationStrategyType] = None
    consistency: Optional[bool] = None
    llm: Optional[LLMConfig] = None


@dataclass
class PrivacyConfig:
    domain: Optional[str] = None
    context_analyzer: Optional[AnalyzerConfig] = None
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    sanitization: SanitizationConfig = field(default_factory=SanitizationConfig)
