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


class AttackVector(str, Enum):
    """Adversarial attack vectors probed by the leakage adversary."""

    RESIDUAL_SPAN = "residual_span"
    QUASI_IDENTIFIER = "quasi_identifier"
    STRUCTURAL_REID = "structural_reid"
    CONTEXT_RECONSTRUCTION = "context_reconstruction"
    MEMBERSHIP_INFERENCE = "membership_inference"


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
class LeakageAdversaryConfig:
    max_iterations: int = 3
    leakage_threshold: float = 0.1
    strategies: List[AttackVector] = field(default_factory=lambda: list(AttackVector))
    llm: Optional[LLMConfig] = None


@dataclass
class PrivacyConfig:
    domain: Optional[str] = None
    interactive: Optional[bool] = None
    context_analyzer: Optional[AnalyzerConfig] = None
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    sanitization: SanitizationConfig = field(default_factory=SanitizationConfig)
    leakage_adversary: LeakageAdversaryConfig = field(
        default_factory=LeakageAdversaryConfig
    )
