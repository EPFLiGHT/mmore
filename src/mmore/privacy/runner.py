"""Drive the compiled privacy graph for a single RAG query."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig

from ..utils import load_config
from .agents.detector import _resolve_engine_tool
from .agents.state import PreCloudOutcome, PrivacyState
from .config import PrivacyConfig
from .domains.profile import get_domain_profile
from .report import ReportRecord
from .verification import VerifierVerdict


@dataclass
class PrivacyResult:
    """Plain result of one privacy-pipeline run, free of raw chunks and PII."""

    answer: str
    record: ReportRecord | None
    verdict: VerifierVerdict | None
    outcome: PreCloudOutcome | None


def validate_privacy_config(config: PrivacyConfig) -> None:
    if config.answer is None:
        raise ValueError("Answer model requires 'answer.llm' in the privacy config.")
    if config.domain:
        get_domain_profile(config.domain)
    if config.detection.engine is not None:
        _resolve_engine_tool(config.detection.engine.value)


def run_privacy_query(
    graph: Any,
    query: str,
    raw_chunks: List[str],
    *,
    request_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> PrivacyResult:
    """Run the privacy pipeline for one query and return its verified result."""
    request_id = request_id or uuid.uuid4().hex
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    thread: RunnableConfig = {"configurable": {"thread_id": request_id}}

    initial = PrivacyState(
        query=query,
        raw_chunks=list(raw_chunks),
        request_id=request_id,
        timestamp=timestamp,
    )
    final = graph.invoke(initial, config=thread)

    if "__interrupt__" in final:
        raise RuntimeError(
            "Privacy gate paused for human approval, which the batch RAG path "
            "cannot handle. Set 'interactive: false' in the privacy config to "
            "auto-decide, or run via the API to approve interactively."
        )

    report = final.get("report") or []
    return PrivacyResult(
        answer=final.get("answer", "") or "",
        record=report[-1] if report else None,
        verdict=final.get("verifier_verdict"),
        outcome=final.get("outcome"),
    )


def load_privacy_config(path: str) -> PrivacyConfig:
    """Load and eagerly validate a ``PrivacyConfig`` from a YAML path."""
    config = load_config(path, PrivacyConfig)
    validate_privacy_config(config)
    return config
