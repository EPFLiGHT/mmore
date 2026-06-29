"""Domain profiles.

A :class:`DomainProfile` parameterizes the privacy pipeline for a given
deployment domain: the sensitive entity set, the per-agent default system
prompts, and the sanitization defaults. There exists three presets: ``global``,
``healthcare`` and ``humanitarian``.
"""

from dataclasses import dataclass
from typing import Dict, List

from ..detection.constants import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_ENTITIES


class UnknownDomainError(KeyError):
    """Raised when a domain name does not match any registered profile."""


@dataclass
class DomainProfile:
    """Per-domain defaults consumed by the privacy agents."""

    name: str
    sensitive_entities: List[str]
    analyzer_system_prompt: str
    sanitizer_system_prompt: str  # TODO: remove as sanitizer can be not llm based
    domain_prompt: str
    default_engine: str = "presidio"
    default_confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    default_strategy: str = "token_masking"
    default_consistency: bool = True


_HEALTHCARE_ENTITIES = [
    "PERSON",
    "MRN",
    "HOSPITAL_DATE",
    "DATE",
    "INSURANCE_ID",
    "SSN",
    "PHONE",
    "EMAIL",
    "LOCATION",
]


PRESIDIO_CLINICAL_PATTERNS = [
    {
        "entity": "MRN",
        "patterns": [
            ("mrn_with_prefix", r"\bMRN[\s:#]*\d{6,10}\b", 0.9),
            ("mrn_bare_8_digits", r"\b\d{8}\b", 0.4),
        ],
        "context": ["mrn", "medical record", "record number", "patient id"],
    },
    {
        "entity": "HOSPITAL_DATE",
        "patterns": [
            ("iso_date", r"\b\d{4}-\d{2}-\d{2}\b", 0.6),
            ("us_date", r"\b\d{1,2}/\d{1,2}/\d{4}\b", 0.6),
        ],
        "context": ["admission", "discharge", "appointment", "hospital", "clinic"],
    },
    {
        "entity": "INSURANCE_ID",
        "patterns": [
            ("insurance_alnum", r"\b[A-Z]{2,3}\d{6,12}\b", 0.7),
        ],
        "context": ["insurance", "policy", "member id", "subscriber"],
    },
]


_HUMANITARIAN_ENTITIES = [
    "PERSON",
    "LOCATION",
    "GPS_COORDINATES",
    "ETHNICITY",
    "LEGAL_STATUS",
    "DISPLACEMENT_STATUS",
    "HOUSEHOLD_ID",
    "PHONE",
    "EMAIL",
    "DATE",
]


GLOBAL_PROFILE = DomainProfile(
    name="global",
    sensitive_entities=list(DEFAULT_ENTITIES),
    analyzer_system_prompt=(
        "You classify the privacy sensitivity of a retrieval-augmented request. "
        "Treat standard personal identifiers as sensitive and flag ambiguous "
        "fields for human review."
    ),
    sanitizer_system_prompt=(
        "Rewrite the text so it carries no personal identifiers while preserving "
        "its factual and topical content."
    ),
    domain_prompt=(
        "General-purpose context: protect personally identifiable information "
        "(names, contact details, identifiers, locations, dates)."
    ),
    default_engine="presidio",  # TODO: After the experiments adjust these
    default_strategy="token_masking",  # TODO: After the experiments adjust these
)

HEALTHCARE_PROFILE = DomainProfile(
    name="healthcare",
    sensitive_entities=list(_HEALTHCARE_ENTITIES),
    analyzer_system_prompt=(
        "You classify the privacy sensitivity of a clinical request. Protected "
        "health information (patient names, medical record numbers, insurance "
        "identifiers, clinical dates) is highly sensitive. Flag ambiguous "
        "clinical fields for human review."
    ),
    sanitizer_system_prompt=(
        "Rewrite the clinical text so it contains no protected health "
        "information while preserving the medical content needed to answer the "
        "request."
    ),
    domain_prompt=(
        "Clinical context: protected health information is highly sensitive; "
        "preserve clinically relevant facts but never patient identity."
    ),
    default_engine="presidio",  # TODO: After the experiments adjust these
    default_strategy="token_masking",  # TODO: After the experiments adjust these
)

HUMANITARIAN_PROFILE = DomainProfile(
    name="humanitarian",
    sensitive_entities=list(_HUMANITARIAN_ENTITIES),
    analyzer_system_prompt=(
        "You classify the privacy sensitivity of a humanitarian request. Data "
        "about affected populations (names, precise locations or GPS, "
        "ethnicity, displacement or legal status, household identifiers) can "
        "endanger people and is highly sensitive. Flag ambiguous fields for "
        "human review."
    ),
    sanitizer_system_prompt=(
        "Rewrite the text so it cannot identify or locate affected individuals "
        "or households while preserving the operational content."
    ),
    domain_prompt=(
        "Humanitarian context: data that could identify or locate affected "
        "people or households is highly sensitive; preserve aggregate "
        "operational facts only."
    ),
    default_engine="presidio",  # TODO: After the experiments adjust these
    default_strategy="token_masking",  # TODO: After the experiments adjust these
)


DOMAIN_PROFILES: Dict[str, DomainProfile] = {
    GLOBAL_PROFILE.name: GLOBAL_PROFILE,
    HEALTHCARE_PROFILE.name: HEALTHCARE_PROFILE,
    HUMANITARIAN_PROFILE.name: HUMANITARIAN_PROFILE,
}


def get_domain_profile(name: str) -> DomainProfile:
    """Return the registered :class:`DomainProfile` for ``name``."""
    try:
        return DOMAIN_PROFILES[name]
    except KeyError:
        raise UnknownDomainError(
            f"Unknown domain '{name}'. Available domains: {sorted(DOMAIN_PROFILES)}"
        ) from None
