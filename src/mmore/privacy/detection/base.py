"""PII detection interface.

Each engine implements ``DetectionEngine.detect`` and returns a list of
``PIISpan`` records. Engines are independently registered as agent tools so a
sanitizer agent can resolve them by name from YAML.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class PIISpan:
    """A single detected PII occurrence in some text."""

    start: int
    end: int
    label: str
    score: float


class DetectionEngine(ABC):
    """Abstract base for PII detection backends."""

    @abstractmethod
    def detect(self, text: str) -> List[PIISpan]:
        """Return all PII spans found in ``text``."""
