"""Drive the compiled privacy graph for a single RAG query.

The thin seam between MMORE's retriever and the privacy pipeline: it seeds the
initial ``PrivacyState`` from the retrieved chunks, runs the compiled graph on a
per-request thread, and returns a plain, PII-free result the RAG chain surfaces.
Raw chunks never leave the state, so they are not logged or returned here.
"""

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
    record: Optional[ReportRecord]
    verdict: Optional[VerifierVerdict]
    outcome: Optional[PreCloudOutcome]


def validate_privacy_config(config: PrivacyConfig) -> None:
    """Surface config errors eagerly, reusing the privacy agents' own errors.

    The graph nodes would otherwise only raise these mid-run, after the first
    query. Checking up front gives the operator a clear failure at startup.
    """
    if config.answer is None:
        raise ValueError("Answer model requires 'answer.llm' in the privacy config.")
    if config.domain:
        get_domain_profile(config.domain)  # raises UnknownDomainError
    if config.detection.engine is not None:
        _resolve_engine_tool(config.detection.engine.value)  # raises KeyError


def run_privacy_query(
    graph: Any,
    query: str,
    raw_chunks: List[str],
    *,
    request_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> PrivacyResult:
    """Run the privacy pipeline for one query and return its verified result.

    In local/batch mode the config sets ``interactive: false``, so the gate
    auto-approves and the graph completes in a single ``invoke``. An interrupt
    here means an interactive config reached the batch path, which it must not.
    """
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
            "Privacy gate interrupted in the batch RAG path. Set 'interactive: "
            "false' in the privacy config, or drive the gate via the API."
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
