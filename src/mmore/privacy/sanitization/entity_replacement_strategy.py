"""Faker-based entity-replacement sanitization strategy.

Substitutes each detected PII span with a realistic fake value drawn from
``faker`` according to the span's label. When ``policy.consistency`` is true,
the same original text always maps to the same fake within the same
``apply`` call.
"""

from typing import TYPE_CHECKING, Callable

from ..agents.registry import register_tool
from ..detection.base import PIISpan
from ..model_cache import MODEL_REGISTRY
from ..schemas.policy import PrivacyPolicy
from .base import SanitizationStrategy, apply_replacements, select_non_overlapping

if TYPE_CHECKING:
    from faker import Faker

_CACHE_KEY = "faker"


def _load_faker() -> "Faker":
    from faker import Faker

    return Faker()


def _fake_value_builders(faker: "Faker") -> dict[str, Callable[[], str]]:
    return {
        "PERSON": faker.name,
        "EMAIL_ADDRESS": faker.email,
        "PHONE_NUMBER": faker.phone_number,
        "DATE_TIME": lambda: faker.date(),
        "HOSPITAL_DATE": lambda: faker.date(),
        "LOCATION": faker.city,
        "GPS_COORDINATES": lambda: f"{faker.latitude()}, {faker.longitude()}",
        "US_SSN": faker.ssn,
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
        chunks: list[str],
        spans_per_chunk: list[list[PIISpan]],
        policy: PrivacyPolicy,
    ) -> list[str]:
        faker = MODEL_REGISTRY.get_or_load(_CACHE_KEY, _load_faker)
        builders = _fake_value_builders(faker)
        consistent = bool(policy.consistency)
        fakes: dict[tuple[str, str], str] = {}

        def fake_for(span: PIISpan, original: str) -> str:
            key = (span.label, original)
            if consistent and key in fakes:
                return fakes[key]
            build = builders.get(span.label)
            fake = str(build()) if build else faker.pystr(min_chars=8, max_chars=12)
            if consistent:
                fakes[key] = fake
            return fake

        return [
            apply_replacements(chunk, select_non_overlapping(spans), fake_for)
            for chunk, spans in zip(chunks, spans_per_chunk)
        ]


@register_tool("sanitize_entity_replacement")
def sanitize_entity_replacement(
    chunks: list[str],
    spans_per_chunk: list[list[PIISpan]],
    policy: PrivacyPolicy,
) -> list[str]:
    """Apply the default-configured entity-replacement strategy."""
    return EntityReplacementStrategy().apply(chunks, spans_per_chunk, policy)
