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


class VerifierCheck(str, Enum):
    """Advisory checks run by the post-cloud verifier over the answer."""

    # TODO: check if we can do other check as well that are more analytic/math based
    RESIDUAL_LEAKAGE = "residual_leakage"
    FAITHFULNESS = "faithfulness"


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
    encryption_key: Optional[str] = None


@dataclass
class LeakageAdversaryConfig:
    enabled: bool = True
    max_iterations: int = 3
    leakage_threshold: float = 0.5
    abort_on_exhaustion: bool = True
    strategies: List[AttackVector] = field(default_factory=lambda: list(AttackVector))
    llm: Optional[LLMConfig] = None


@dataclass
class CloudLLMConfig:
    llm: LLMConfig
    system_prompt: Optional[str] = None


@dataclass
class VerifierConfig:
    checks: List[VerifierCheck] = field(default_factory=lambda: list(VerifierCheck))
    warn_threshold: float = 0.5
    llm: Optional[LLMConfig] = None


@dataclass
class PrivacyConfig:
    domain: Optional[str] = None
    interactive: bool = False
    context_analyzer: Optional[AnalyzerConfig] = None
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    sanitization: SanitizationConfig = field(default_factory=SanitizationConfig)
    leakage_adversary: LeakageAdversaryConfig = field(
        default_factory=LeakageAdversaryConfig
    )
    answer: Optional[CloudLLMConfig] = None
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
