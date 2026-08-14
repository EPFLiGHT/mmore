"""Shared defaults for the PII detection engines."""

from dataclasses import dataclass

from ...rag.llm import LLMConfig

DEFAULT_LANGUAGE = "en"

DEFAULT_GLINER_MODEL = "nvidia/gliner-PII"
DEFAULT_OPENAI_FILTER_MODEL = "openai/privacy-filter"
DEFAULT_PRESIDIO_SPACY_MODEL = "en_core_web_lg"

DEFAULT_LLM_CONFIG = LLMConfig(
    llm_name="Qwen/Qwen2.5-3B-Instruct",
    max_new_tokens=512,
)

THRESHOLD_LEVELS: dict[str, float] = {
    "low": 0.5,
    "medium": 0.7,
    "high": 0.85,
}

DEFAULT_CONFIDENCE_THRESHOLD = THRESHOLD_LEVELS["medium"]


def threshold_or_default(value: float | None) -> float:
    return DEFAULT_CONFIDENCE_THRESHOLD if value is None else value


DEFAULT_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "MRN",
    "DATE_TIME",
    "LOCATION",
    "US_SSN",
    "INSURANCE_ID",
]


# Custom clinical recognizers added on top of Presidio's built-in ones
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


# Short engine names used in YAML configs mapped to the tool names
DETECTION_TOOL_NAMES = {
    "presidio": "detect_pii_presidio",
    "gliner": "detect_pii_gliner",
    "openai_filter": "detect_pii_openai_filter",
    "llm": "detect_pii_llm",
}


# Per-engine guidance for the analyzer's engine selector
DETECTION_GUIDANCE: dict[str, str] = {
    "presidio": (
        "Presidio: rule-based detection + spaCy NER, augmented with the "
        "clinical recognizers shipped. Precise but cautious, and the weakest of "
        "the precise engines. Good on common structured identifiers, poor on "
        "rare free-text attributes. Pick it when the text is well-formatted and "
        "predictable, or when no model can be loaded."
    ),
    "gliner": (
        "GLiNER: zero-shot transformer NER over an arbitrary label set "
        "(default: nvidia/gliner-PII). Precise on the entity types it knows "
        "well, but cautious, and it degrades on rare attribute types and on "
        "text unlike its training data. Pick it when false positives are costly "
        "and the expected entity types are common."
    ),
    "openai_filter": (
        "openai/privacy-filter: HuggingFace token-classification model from "
        "OpenAI for PII. Trades precision for recall: its token-level tagger "
        "over-predicts and splits each identifier into sub-word spans, so its "
        "output needs merging. Strong on names, dates and account numbers, weak "
        "on attributes expressed in words rather than in a fixed pattern. Pick "
        "it when a miss is worse than a false alarm."
    ),
    "llm": (
        "LLM-backed detection via DSPy typed structured output (uses the "
        "configured LLM with a constrained schema). The most even across entity "
        "types, and the one that holds up best on the rare, domain-specific "
        "identifiers the other engines miss. Slowest and costliest. Pick it for "
        "heterogeneous text, and whenever the user states very specific "
        "detection needs: it is the only engine that can be steered by the "
        "request itself."
    ),
}


@dataclass
class DetectionParams:
    """Tunable parameter every engine accepts."""

    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD


@dataclass
class GLiNERParams(DetectionParams):
    multi_label: bool = False


DETECTION_DEFAULT_PARAMS: dict[str, DetectionParams] = {
    "presidio": DetectionParams(),
    "gliner": GLiNERParams(),
    "openai_filter": DetectionParams(),
    "llm": DetectionParams(),
}

_CONFIDENCE_THRESHOLD_GUIDANCE = (
    "- confidence_threshold: how strict we are when deciding whether a span truly deserves a label.\n"
    "  - low: favor recall, include weak or implicit signals.\n"
    "  - medium: balanced, label when reasoning is plausible.\n"
    "  - high: favor precision, only accept well-justified matches."
)

# Per-engine guidance for the analyzer's engine parameter selector
DETECTION_PARAM_GUIDANCE: dict[str, str] = {
    "presidio": (
        "Presidio (rule-based + spaCy NER + clinical recognizers).\n"
        f"{_CONFIDENCE_THRESHOLD_GUIDANCE}"
    ),
    "gliner": (
        "GLiNER (zero-shot NER over arbitrary labels).\n"
        f"{_CONFIDENCE_THRESHOLD_GUIDANCE}\n"
        "- multi_label: true allows overlapping labels; false picks one label per span."
    ),
    "openai_filter": (
        "openai/privacy-filter (HF token-classification).\n"
        f"{_CONFIDENCE_THRESHOLD_GUIDANCE}"
    ),
    "llm": (
        "LLM-backed detection (DSPy structured output).\n"
        f"{_CONFIDENCE_THRESHOLD_GUIDANCE}"
    ),
}
