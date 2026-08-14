"""Drive the compiled privacy graph for a single RAG query."""

import logging
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from ..utils import load_config
from .agents.detector import _resolve_engine_tool
from .agents.state import PrivacyState
from .config import PrivacyConfig
from .domains import get_domain_profile
from .gate_ui import Approver, make_terminal_approver
from .schemas.report import PreCloudOutcome, ReportRecord
from .schemas.verification import VerifierVerdict

logger = logging.getLogger(__name__)


@dataclass
class PrivacyResult:
    """What one run of the privacy pipeline returns, minus the raw chunks."""

    answer: str
    record: ReportRecord | None
    verdict: VerifierVerdict | None
    outcome: PreCloudOutcome | None
    sanitized_chunks: list[str]


def validate_privacy_config(config: PrivacyConfig) -> None:
    """Fail now, not mid-query, on a config the pipeline cannot run."""
    if config.answer is None:
        raise ValueError("Answer model requires 'answer.llm' in the privacy config.")
    if config.domain:
        get_domain_profile(config.domain)
    if config.detection.engine is not None:
        _resolve_engine_tool(config.detection.engine.value)


def load_privacy_config(path: str) -> PrivacyConfig:
    """Load a privacy config from a YAML path and check it right away."""
    config = load_config(path, PrivacyConfig)
    validate_privacy_config(config)
    return config


def setup_privacy(
    config: PrivacyConfig | str,
    *,
    interactive_ok: bool = True,
    review_card: bool = True,
) -> tuple[CompiledStateGraph, Approver | None, PrivacyConfig]:
    """Compile the graph from a config, or from the path to one."""
    from langgraph.checkpoint.memory import MemorySaver

    from .pipeline import build_privacy_pipeline

    if isinstance(config, PrivacyConfig):
        validate_privacy_config(config)
    else:
        config = load_privacy_config(config)

    approver: Approver | None = None
    if config.interactive:
        if interactive_ok:
            approver = make_terminal_approver(review_card=review_card)
        else:
            logger.warning(
                "The interactive privacy gate needs a terminal."
                "Set 'interactive: false' to remove this warning."
            )
            config = replace(config, interactive=False)

    return build_privacy_pipeline(config, MemorySaver()), approver, config


def run_privacy_query(
    graph: CompiledStateGraph,
    query: str,
    raw_chunks: list[str],
    *,
    request_id: str | None = None,
    timestamp: str | None = None,
    approver: Approver | None = None,
) -> PrivacyResult:
    """Run the privacy pipeline for one query and return its verified result."""
    request_id = request_id or uuid.uuid4().hex
    thread: RunnableConfig = {"configurable": {"thread_id": request_id}}

    final = graph.invoke(
        PrivacyState(
            query=query,
            raw_chunks=list(raw_chunks),
            request_id=request_id,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        ),
        config=thread,
    )

    while "__interrupt__" in final:
        if approver is None:
            raise RuntimeError(
                "Privacy gate paused for human approval but no approver is "
                "available on this path. Set 'interactive: false' in the "
                "privacy config to auto-decide."
            )
        payload = final["__interrupt__"][0].value
        final = graph.invoke(Command(resume=approver(payload)), config=thread)

    report = final.get("report") or []
    return PrivacyResult(
        answer=final.get("answer", "") or "",
        record=report[-1] if report else None,
        verdict=final.get("verifier_verdict"),
        outcome=final.get("outcome"),
        sanitized_chunks=list(final.get("sanitized_chunks", [])),
    )
