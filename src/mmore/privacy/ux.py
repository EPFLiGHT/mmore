"""Stage reporting for privacy mode.

The privacy pipeline runs inside a single RAG query, so it owns no display of
its own: each graph node announces the agent it is about to run and the RAG
runner renders it on the surface it owns (a shared bar, or a live status).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Union

from rich.markup import escape

from ..ux import Color

PRIVACY_NAME = "Privacy"
PRIVACY_EMOJI = "🛡"

# The one stage that is not part of the privacy pipeline
ANSWER_STAGE = "Answer"

FLOW = (
    "Analyzer",
    "Detector",
    "Sanitizer",
    "Adversary",
    "Gate",
    ANSWER_STAGE,
    "Verifier",
)

_TOOL_NAMES = {
    "gliner": "GLiNER",
    "llm": "Prompted SLM",
    "openai_filter": "OpenAI filter",
    "presidio": "Presidio",
    "token_masking": "Token masking",
    "entity_replacement": "Entity replacement",
    "synthetic_rewrite": "Synthetic rewrite",
}


def tool_name(tool: Union[str, Enum]) -> str:
    value: str = tool.value if isinstance(tool, Enum) else tool
    return _TOOL_NAMES.get(value, value.replace("_", " ").capitalize())


@dataclass(frozen=True)
class Stage:
    """One privacy agent, as shown to the user while it runs."""

    agent: str
    action: str
    unit: str


StageCallback = Callable[[Stage], None]
NoticeCallback = Callable[[str], None]

_stage_callback: Optional[StageCallback] = None
_notice_callback: Optional[NoticeCallback] = None


def set_stage_callback(callback: Optional[StageCallback]) -> None:
    """Route the pipeline's stages to a renderer."""
    global _stage_callback
    _stage_callback = callback


def report_stage(stage: Stage) -> None:
    """Announce the agent the pipeline is about to run."""
    if _stage_callback is not None:
        _stage_callback(stage)


def set_notice_callback(callback: Optional[NoticeCallback]) -> None:
    """Route the pipeline's one-off notices to a renderer."""
    global _notice_callback
    _notice_callback = callback


def report_notice(message: str) -> bool:
    """Surface a one-off event (a leak, a revision)."""
    if _notice_callback is None:
        return False
    _notice_callback(message)
    return True


def _draw_stage(surface: Any, stage: Stage) -> None:
    """Draw the privacy graph with the running agent highlighted, and what it does."""
    nodes = [
        f"[bold {Color.MMORE}]\\[{name}][/]"
        if name == stage.agent
        else f"[dim]{name}[/]"
        for name in FLOW
    ]
    graph = "[dim] → [/]".join(nodes)
    action = escape(stage.action[0].upper() + stage.action[1:])
    surface.set_status(
        f"  [{Color.ACCENT}]Privacy pipeline:[/] {graph}",
        f"  [dim]{action}...[/]",
    )
    surface.set_unit(stage.unit)


def attach(surface: Any) -> None:
    """Render the pipeline's stages and notices on a progress bar or a status."""
    set_stage_callback(lambda stage: _draw_stage(surface, stage))
    set_notice_callback(
        lambda message: surface.print_above(
            f"  [{Color.ORANGE}]⚠ {escape(message)}[/]",
        )
    )


def detach() -> None:
    set_stage_callback(None)
    set_notice_callback(None)
