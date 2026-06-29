"""Coarse risk assessment produced by the Detector.

A cheap summary of how much sensitive material a request's retrieved context
carries: how many spans, of which types, and how dense.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RiskAssessment:
    """Aggregate sensitivity signal over the detected spans."""

    count: int
    entity_counts: Dict[str, int] = field(default_factory=dict)
    density: float = 0.0
    level: str = "low"
