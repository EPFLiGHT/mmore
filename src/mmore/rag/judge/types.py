"""Judge types: decisions, results, and configuration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

from ..llm import LLMConfig

_CORE_METRIC_FIELDS = (
    "num_docs",
    "mean_similarity",
    "max_similarity",
    "mean_rerank_score",
    "max_rerank_score",
)


@dataclass
class RetrievalMetrics:
    num_docs: float
    mean_similarity: float
    max_similarity: float
    mean_rerank_score: float
    max_rerank_score: float
    context_relevance_score: Optional[float] = None
    thresholds_met: Optional[float] = None

    def threshold_items(self) -> Dict[str, float]:
        items = self.core_dict()
        if self.context_relevance_score is not None:
            items["context_relevance_score"] = self.context_relevance_score
        return items

    def core_dict(self) -> Dict[str, float]:
        return {field: getattr(self, field) for field in _CORE_METRIC_FIELDS}

    def to_dict(self) -> Dict[str, float]:
        data = self.threshold_items()
        if self.thresholds_met is not None:
            data["thresholds_met"] = self.thresholds_met
        return data


@dataclass
class CorrectionRecord:
    action: str
    before: RetrievalMetrics
    after: RetrievalMetrics
    thresholds_met_before: float
    thresholds_met_after: float

    def delta_dict(self) -> Dict[str, float]:
        return {
            f"delta_{field}": getattr(self.after, field) - getattr(self.before, field)
            for field in _CORE_METRIC_FIELDS
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "before": self.before.core_dict(),
            "after": self.after.core_dict(),
            "delta": self.delta_dict(),
            "thresholds_met_before": self.thresholds_met_before,
            "thresholds_met_after": self.thresholds_met_after,
        }


# Fields added to RAG pipeline state by retrieve_with_judge (public JSON/API output).
JUDGE_OUTPUT_KEYS = (
    "judge_decision",
    "judge_reason",
    "judge_actions",
    "judge_llm_calls",
    "judge_steps",
    "hit_max_corrective_steps",
    "retrieval_metrics",
    "retrieval_corrections",
)


def extract_judge_output(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Pick judge trace fields for public RAG output (JSON export / API)."""
    return {
        key: state[key]
        for key in JUDGE_OUTPUT_KEYS
        if key in state and state[key] is not None
    }


class JudgeDecision(str, Enum):
    PROCEED = "PROCEED"
    ADD_QUESTIONS = "ADD_QUESTIONS"
    ADD_CONTEXT = "ADD_CONTEXT"
    RE_RETRIEVE = "RE_RETRIEVE"


@dataclass
class JudgeResult:
    decision: JudgeDecision
    reason: str = ""
    exit_reason: str = ""
    context_relevance_score: Optional[float] = None
    extra_questions: List[str] = field(default_factory=list)
    web_query: Optional[str] = None
    retrieve_params: Optional[Dict[str, Any]] = None
    llm_invoked: bool = False
    raw_llm_response: Optional[str] = None
    raw_decision: Optional[str] = None
    coerced_decision: bool = False

    @classmethod
    def proceed(
        cls,
        reason: str,
        *,
        llm_invoked: bool = False,
        **kwargs: Any,
    ) -> "JudgeResult":
        return cls(
            decision=JudgeDecision.PROCEED,
            reason=reason,
            exit_reason=reason,
            llm_invoked=llm_invoked,
            **kwargs,
        )


@dataclass
class JudgeConfig:
    """LLM judge config. Only LLMJudge is implemented."""

    llm: LLMConfig
    system_prompt: str
    user_prompt: str
    metric_thresholds: Dict[str, float] = field(default_factory=dict)
    max_corrective_steps: int = 1
    allow_add_questions: bool = True
    allow_add_context: bool = True
    allow_re_retrieve: bool = True
    rerank_after_merge: bool = True
    max_web_results: int = 5
    max_chunk_chars: int = 400
    force_corrective_action: Optional[str] = None
