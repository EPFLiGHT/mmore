"""Top-level configuration for the privacy pipeline."""

from dataclasses import dataclass, field
from enum import Enum

from ..rag.llm import LLMConfig
from ..utils import load_config


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

    RESIDUAL_LEAKAGE = "residual_leakage"
    FAITHFULNESS = "faithfulness"


@dataclass
class AnalyzerConfig:
    llm: LLMConfig
    system_prompt: str | None = None


@dataclass
class DetectionConfig:
    engine: DetectionEngineType | None = None
    confidence_threshold: float | None = None
    entity_types: list[str] = field(default_factory=list)
    llm: LLMConfig | None = None


@dataclass
class SanitizationConfig:
    strategy: SanitizationStrategyType | None = None
    consistency: bool | None = None
    llm: LLMConfig | None = None
    encryption_key: str | None = None


@dataclass
class LeakageAdversaryConfig:
    enabled: bool = True
    max_iterations: int = 3
    leakage_threshold: float = 0.5
    abort_on_exhaustion: bool = True
    strategies: list[AttackVector] = field(default_factory=lambda: list(AttackVector))
    llm: LLMConfig | None = None


@dataclass
class CloudLLMConfig:
    llm: LLMConfig
    system_prompt: str | None = None


@dataclass
class VerifierConfig:
    checks: list[VerifierCheck] = field(default_factory=lambda: list(VerifierCheck))
    warn_threshold: float = 0.5
    llm: LLMConfig | None = None


@dataclass
class PrivacyConfig:
    domain: str | None = None
    interactive: bool = False
    context_analyzer: AnalyzerConfig | None = None
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    sanitization: SanitizationConfig = field(default_factory=SanitizationConfig)
    leakage_adversary: LeakageAdversaryConfig = field(
        default_factory=LeakageAdversaryConfig
    )
    answer: CloudLLMConfig | None = None
    verifier: VerifierConfig = field(default_factory=VerifierConfig)


def as_privacy_config(config: PrivacyConfig | str | dict) -> PrivacyConfig:
    """Accept an already-built config, or the YAML path/mapping to load it from."""
    if isinstance(config, PrivacyConfig):
        return config
    return load_config(config, PrivacyConfig)
