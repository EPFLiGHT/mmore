"""Structured output of the post-cloud Advisory Verifier.

Emitted by the VerifierAgent after checking the model's answer against
the whole context (raw + sanitized). It never re-triggers an escalation.
"""

from collections import Counter
from dataclasses import dataclass, field

from ..config import VerifierCheck


@dataclass
class VerifierWarning:
    """One advisory finding from a single check over the answer."""

    kind: VerifierCheck
    # entity type (if residual leakage) or short unsupported claim (if faithfulness)
    flagged: str | None
    evidence: str
    confidence: float


@dataclass
class VerifierVerdict:
    """Aggregate advisory verdict: the warnings raised across all checks."""

    warnings: list[VerifierWarning] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.warnings

    @property
    def summary(self) -> str:
        if not self.warnings:
            return "clean"
        counts = Counter(w.kind.value for w in self.warnings)
        breakdown = ", ".join(f"{kind}: {n}" for kind, n in sorted(counts.items()))
        return f"{len(self.warnings)} warning(s) ({breakdown})"


# Verdict for an answer that passed both checks
CLEAN_VERDICT = VerifierVerdict(warnings=[])
