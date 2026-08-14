"""Coarse risk assessment produced by the Detector."""

from dataclasses import dataclass, field


@dataclass
class RiskAssessment:
    """Aggregate sensitivity signal over the detected spans."""

    count: int
    entity_counts: dict[str, int] = field(default_factory=dict)
    density: float = 0.0
    level: str = "low"
