"""Bounded escalation of the privacy policy for the leakage loop.

When the leakage adversary flags the sanitized context, the loop re-enters the
Detector and Sanitizer under a stricter policy. ``escalate_policy`` returns a
new, more aggressive ``PrivacyPolicy`` (it never mutates its input) plus a
short label of what changed, for the escalation log read by PR #7.

The escalation ladder follows the spec: merge the analyzer's leak-targeted
entity labels, then lower the detection threshold, then bring in a detection
engine not used on the previous pass, then switch to a stricter sanitization
strategy (masking -> synthetic rewrite). Steps apply cumulatively because each
call receives the policy already escalated by the previous iterations.
"""

from dataclasses import replace
from typing import Callable, List, Optional, Tuple

from .policy import PrivacyPolicy

_CONFIDENCE_FLOOR = 0.1
_THRESHOLD_DECAY = 0.5

# Detection-engine ladder (might change after experiments)
_ENGINE_LADDER = ["presidio", "gliner", "llm"]

# Stricter-sanitization ladder (might change after experiments)
_STRATEGY_LADDER = ["token_masking", "entity_replacement", "synthetic_rewrite"]

_ESCALATION_NOTE = (
    "Escalation: a prior sanitization pass leaked: treat every quasi-identifier "
    "and rare attribute as sensitive and redact aggressively."
)


def _next_in_ladder(ladder: List[str], current: str, cap: bool) -> str:
    """Next entry after ``current``, cap at the last entry or wrap around."""
    if current not in ladder:
        return ladder[0]
    nxt = ladder.index(current) + 1
    if nxt >= len(ladder):
        return ladder[-1] if cap else ladder[0]
    return ladder[nxt]


def _broaden_entities(
    policy: PrivacyPolicy, extra_entities: Optional[List[str]]
) -> PrivacyPolicy:
    """Merge the analyzer's leak-targeted labels and pin the escalation note."""
    entities = list(policy.sensitive_entities) + [
        e for e in (extra_entities or []) if e not in policy.sensitive_entities
    ]
    prompt = policy.domain_prompt
    if _ESCALATION_NOTE not in prompt:
        prompt = prompt + _ESCALATION_NOTE
    return replace(policy, sensitive_entities=entities, domain_prompt=prompt)


def _lower_threshold(policy: PrivacyPolicy) -> Tuple[PrivacyPolicy, str]:
    params = dict(policy.detection_params)
    current = float(params.get("confidence_threshold", 0.7))
    params["confidence_threshold"] = max(
        _CONFIDENCE_FLOOR, round(current * _THRESHOLD_DECAY, 2)
    )
    return replace(policy, detection_params=params), "lower_threshold"


def _switch_detection_engine(policy: PrivacyPolicy) -> Tuple[PrivacyPolicy, str]:
    nxt = _next_in_ladder(_ENGINE_LADDER, policy.detection_engine, cap=False)
    return replace(policy, detection_engine=nxt), f"switch_engine_to_{nxt}"


def _stricter_sanitization(policy: PrivacyPolicy) -> Tuple[PrivacyPolicy, str]:
    nxt = _next_in_ladder(_STRATEGY_LADDER, policy.sanitization_strategy, cap=True)
    return replace(policy, sanitization_strategy=nxt), f"stricter_strategy_{nxt}"


_STEPS: List[Callable[[PrivacyPolicy], Tuple[PrivacyPolicy, str]]] = [
    _lower_threshold,
    _switch_detection_engine,
    _stricter_sanitization,
]


def escalate_policy(
    policy: PrivacyPolicy,
    iteration: int,
    extra_entities: Optional[List[str]] = None,
) -> Tuple[PrivacyPolicy, str]:
    """Return a stricter ``PrivacyPolicy`` and a label for the escalation log.

    Each round merges the analyzer's leak-targeted ``extra_entities`` (if any)
    and applies one progressive knob: lower threshold, then switch detection
    engine, then stricter sanitization.
    """
    broadened = _broaden_entities(policy, extra_entities)
    step = _STEPS[min(iteration, len(_STEPS) - 1)]
    return step(broadened)
