"""Structured output of the post-cloud Advisory Verifier.

Emitted by the VerifierAgent after checking the model's answer against
the whole context (raw + sanitized). It never re-triggers detection or
sanitization.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class WarningKind(str, Enum):
    """The advisory check that raised a warning."""

    RESIDUAL_LEAKAGE = "residual_leakage"
    FAITHFULNESS = "faithfulness"


@dataclass
class VerifierWarning:
    """One advisory finding from a single check over the answer."""

    kind: WarningKind
    # entity type (if residual leakage) or short unsupported claim (if faithfulness)
    flagged: Optional[str]
    evidence: str
    confidence: float


@dataclass
class VerifierVerdict:
    """Aggregate advisory verdict: the warnings raised across all checks."""

    warnings: List[VerifierWarning] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.warnings

    @property
    def summary(self) -> str:
        if not self.warnings:
            return "clean"
        counts = {}
        for w in self.warnings:
            counts[w.kind.value] = counts.get(w.kind.value, 0) + 1
        breakdown = ", ".join(f"{kind}: {n}" for kind, n in sorted(counts.items()))
        return f"{len(self.warnings)} warning(s) ({breakdown})"


# Verdict for an answer that passed both checks
CLEAN_VERDICT = VerifierVerdict(warnings=[])
