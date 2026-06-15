"""Bounded escalation of the privacy policy for the leakage loop.

When the leakage adversary flags the sanitized context, the loop re-enters the
Detector and Sanitizer under a stricter policy. Returns a new and stricter
PrivacyPolicy plus a short label of what changed."""

from dataclasses import replace
from typing import Callable, List, Optional, Tuple

from .config import DetectionEngineType, SanitizationStrategyType
from .policy import PrivacyPolicy

_CONFIDENCE_FLOOR = 0.1
_THRESHOLD_DECAY = 0.5

# Detection-engine ladder (might change after experiments)
_ENGINE_LADDER = [
    DetectionEngineType.PRESIDIO.value,
    DetectionEngineType.GLINER.value,
    DetectionEngineType.LLM.value,
]

# Stricter-sanitization ladder (might change after experiments)
_STRATEGY_LADDER = [
    SanitizationStrategyType.TOKEN_MASKING.value,
    SanitizationStrategyType.ENTITY_REPLACEMENT.value,
    SanitizationStrategyType.SYNTHETIC_REWRITE.value,
]

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


def apply_entity_guidance(
    policy: PrivacyPolicy,
    extra_entities: Optional[List[str]] = None,
    context_note: Optional[str] = None,
) -> PrivacyPolicy:
    """Merge targeted entity labels and pin the escalation note plus any human
    guidance into the domain prompt."""
    entities = list(policy.sensitive_entities) + [
        e for e in (extra_entities or []) if e not in policy.sensitive_entities
    ]
    prompt = policy.domain_prompt
    if _ESCALATION_NOTE not in prompt:
        prompt = prompt + _ESCALATION_NOTE
    if context_note:
        guidance = f" Human guidance: {context_note}"
        if guidance not in prompt:
            prompt = prompt + guidance
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
    """Return a stricter ``PrivacyPolicy`` and a label for the escalation log."""
    broadened = apply_entity_guidance(policy, extra_entities)
    step = _STEPS[min(iteration, len(_STEPS) - 1)]
    return step(broadened)
