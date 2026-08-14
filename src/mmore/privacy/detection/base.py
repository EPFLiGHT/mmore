"""PII detection interface.

Each engine implements ``DetectionEngine.detect`` and returns a list of
``PIISpan`` records. Engines are independently registered as agent tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from ..schemas.policy import PrivacyPolicy


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
    def detect(self, text: str) -> list[PIISpan]:
        """Return all PII spans found in ``text``."""


# A registered detection tool: scan one text under a policy
DetectionTool = Callable[[str, PrivacyPolicy], list[PIISpan]]
