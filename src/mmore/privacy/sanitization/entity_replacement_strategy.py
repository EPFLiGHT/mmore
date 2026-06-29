"""Faker-based entity-replacement sanitization strategy.

Substitutes each detected PII span with a realistic fake value drawn from
``faker`` according to the span's label. When ``policy.consistency`` is true,
the same original text always maps to the same fake within the same
``apply`` call.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

from .._cache import MODEL_REGISTRY
from ..agents.registry import register_tool
from ..detection.base import PIISpan
from ..policy import PrivacyPolicy
from .base import SanitizationStrategy, apply_replacements, select_non_overlapping

if TYPE_CHECKING:
    from faker import Faker

_CACHE_PREFIX = "faker"


def _load_faker() -> "Faker":
    from faker import Faker

    return Faker()


def clear_faker_cache() -> None:
    """Drop the cached Faker instance."""
    MODEL_REGISTRY.clear(prefix=_CACHE_PREFIX)


def _build_label_map(faker: "Faker") -> Dict[str, Callable[[], str]]:
    return {
        "PERSON": faker.name,
        "EMAIL": faker.email,
        "PHONE": faker.phone_number,
        "DATE": lambda: faker.date(),
        "HOSPITAL_DATE": lambda: faker.date(),
        "LOCATION": faker.city,
        "GPS_COORDINATES": lambda: f"{faker.latitude()}, {faker.longitude()}",
        "SSN": faker.ssn,
        "MRN": lambda: f"MRN{faker.random_number(digits=8, fix_len=True)}",
        "INSURANCE_ID": lambda: faker.bothify(text="??########").upper(),
        "ETHNICITY": faker.word,
        "LEGAL_STATUS": faker.word,
        "DISPLACEMENT_STATUS": faker.word,
        "HOUSEHOLD_ID": lambda: faker.bothify(text="HH########").upper(),
    }


class EntityReplacementStrategy(SanitizationStrategy):
    """Replace each span with a Faker-generated fake value for its label."""

    def apply(
        self,
        chunks: List[str],
        spans_per_chunk: List[List[PIISpan]],
        policy: PrivacyPolicy,
    ) -> List[str]:
        faker = MODEL_REGISTRY.get_or_load(_CACHE_PREFIX, _load_faker)
        label_map = _build_label_map(faker)
        consistency = bool(policy.consistency)
        memo: Dict[Tuple[str, str], str] = {}

        def fake_for_label(label: str) -> str:
            gen = label_map.get(label)
            if gen is None:
                return faker.pystr(min_chars=8, max_chars=12)
            return str(gen())

        def replace(span: PIISpan, original: str) -> str:
            if consistency:
                key = (span.label, original)
                cached = memo.get(key)
                if cached is not None:
                    return cached
                fake = fake_for_label(span.label)
                memo[key] = fake
                return fake
            return fake_for_label(span.label)

        out: List[str] = []
        for chunk, spans in zip(chunks, spans_per_chunk):
            kept = select_non_overlapping(list(spans))
            out.append(apply_replacements(chunk, kept, replace))
        return out


@register_tool("sanitize_entity_replacement")
def sanitize_entity_replacement(
    chunks: List[str],
    spans_per_chunk: List[List[PIISpan]],
    policy: PrivacyPolicy,
) -> List[str]:
    """Apply the default-configured entity-replacement strategy."""
    return EntityReplacementStrategy().apply(chunks, spans_per_chunk, policy)
