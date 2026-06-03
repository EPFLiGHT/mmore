"""Shared defaults for the PII detection engines."""

from ...rag.llm import LLMConfig

DEFAULT_LANGUAGE = "en"

DEFAULT_GLINER_MODEL = "nvidia/gliner-PII"
DEFAULT_OPENAI_FILTER_MODEL = "openai/privacy-filter"
DEFAULT_PRESIDIO_SPACY_MODEL = "en_core_web_lg"

DEFAULT_LLM_CONFIG = LLMConfig(
    llm_name="Qwen/Qwen2.5-3B-Instruct",
    max_new_tokens=512,
)

DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# TODO: Later add new labels to the list
DEFAULT_LABELS = [
    "PERSON",
    "PHONE",
    "EMAIL",
    "MRN",
    "DATE",
    "LOCATION",
    "SSN",
    "INSURANCE_ID",
]

# TODO: Later add new patterns to the list
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
