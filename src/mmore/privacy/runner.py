"""Drive the compiled privacy graph for a single RAG query."""

import difflib
import importlib.util
import logging
import math
import re
import shutil
import sys
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Tuple, Union

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from ..utils import load_config
from ..ux import Color, card, paused_progress, str_brand, str_in_color
from .agents.detector import _resolve_engine_tool
from .agents.state import PreCloudOutcome, PrivacyState
from .config import PrivacyConfig
from .domains.profile import get_domain_profile
from .report import ReportRecord
from .verification import VerifierVerdict

logger = logging.getLogger(__name__)


@dataclass
class PrivacyResult:
    """What one run of the privacy pipeline returns, minus the raw chunks."""

    answer: str
    record: ReportRecord | None
    verdict: VerifierVerdict | None
    outcome: PreCloudOutcome | None
    sanitized_chunks: List[str]


# Answers one gate interrupt and resumes the graph
Approver = Callable[[dict], object]

_VIEW_CHOICES = {"v", "view"}

_RED, _RESET = "\033[31m", "\033[0m"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_FALLBACK_SIZE = (100, 24)
_GATE_CARD_TITLE = "Privacy 🛡 · review before the cloud call"
_VIEW_OPTION_LABEL = (
    "View raw vs sanitized context diff (flagged PII in red, replacements in yellow)"
)
_VIEW_LEGEND = "(red = flagged PII, yellow = the replacement)"


def _gate_choice(payload: dict, action: str) -> str:
    """The menu number the gate gave an action, falling back to the action name."""
    for option in payload.get("options", []):
        if option.get("action") == action:
            return str(option["choice"])
    return action


def _render_chunk_diff(raw: str, sanitized: str) -> str:
    """Word-level diff of one chunk: what was removed in red, what replaced it in mmore's color."""
    raw_words = re.split(r"(\s+)", raw)
    sanitized_words = re.split(r"(\s+)", sanitized)
    opcodes = difflib.SequenceMatcher(
        None, raw_words, sanitized_words, autojunk=False
    ).get_opcodes()

    parts: List[str] = []
    for op, raw_lo, raw_hi, san_lo, san_hi in opcodes:
        if op == "equal":
            parts.append("".join(sanitized_words[san_lo:san_hi]))
            continue
        removed = "".join(raw_words[raw_lo:raw_hi]).strip()
        inserted = "".join(sanitized_words[san_lo:san_hi]).strip()
        # The trailing space keeps colored segments apart
        if removed:
            parts.append(f"{_RED}{removed}{_RESET} ")
        if inserted:
            parts.append(f"{str_brand(inserted)} ")
    return "".join(parts).strip()


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _wrap_ansi(line: str, width: int) -> List[str]:
    """Wrap to `width` visible columns, keeping the current color across the breaks."""
    wrapped: List[str] = []
    words: List[str] = []
    length = 0
    active = ""

    def flush() -> None:
        text = " ".join(words)
        wrapped.append(f"{active}{text}{_RESET}" if active else text)

    for word in line.split(" "):
        size = _visible_len(word)
        if words and length + 1 + size > width:
            flush()
            words, length = [], 0
        length += size + (1 if words else 0)
        words.append(word)
        codes = _ANSI_RE.findall(word)
        if codes:
            active = "" if codes[-1] == _RESET else codes[-1]
    flush()
    return wrapped


def _quoted_block(title: str, body: str) -> str:
    """A title and its body, with a colored bar down the left margin."""
    bar = str_brand("│")
    width = max(40, shutil.get_terminal_size(_FALLBACK_SIZE).columns - 2)
    lines = [f"{bar} {str_brand(title, bold=True)}", bar]
    for line in body.splitlines():
        lines += [f"{bar} {wrapped}" for wrapped in _wrap_ansi(line, width)]
    return "\n".join(lines)


def render_chunks(query: Optional[dict], chunks: List[dict]) -> str:
    """The raw text next to the sanitized one, for the query and every chunk."""
    if not query and not chunks:
        return "No sanitized context available."
    blocks = [str_in_color(_VIEW_LEGEND, Color.GRAY)]
    if query:
        diff = _render_chunk_diff(query.get("raw", ""), query.get("sanitized", ""))
        blocks.append(_quoted_block("Query", diff))
    for number, chunk in enumerate(chunks, 1):
        diff = _render_chunk_diff(chunk.get("raw", ""), chunk.get("sanitized", ""))
        blocks.append(_quoted_block(f"Chunk {number}", diff))
    return "\n\n".join(blocks)


def _print_review(payload: dict) -> None:
    """Show what the pipeline did, as a card when the gate sent the details."""
    details = payload.get("details")
    if details:
        card(_GATE_CARD_TITLE, {d["label"]: d["value"] for d in details})
    else:
        print(payload.get("summary", ""))


def _print_menu(payload: dict) -> None:
    for option in payload.get("options", []):
        choice = str_brand(f"[{option['choice']}]", bold=True)
        print(f"  {choice} {option['label']}")
    print(f"  {str_brand('[v]', bold=True)} {_VIEW_OPTION_LABEL}")


def _show_sanitized(payload: dict) -> None:
    print()
    print(render_chunks(payload.get("query"), payload.get("chunks", [])))
    print()


def _erase_printed(text: str) -> None:
    """Erase what `print(text)` just drew, including the lines it wrapped onto."""
    if not sys.stdout.isatty():
        return
    width = max(1, shutil.get_terminal_size(_FALLBACK_SIZE).columns)
    rows = max(1, math.ceil(_visible_len(text) / width))
    sys.stdout.write(f"\033[{rows}F\033[J")
    sys.stdout.flush()


def _menu_gate(payload: dict, banner: Optional[str] = None) -> object:
    """Arrow-key gate menu. `banner` is redrawn above each prompt and erased with it."""
    import questionary

    from ..tui.theme import QMARK, QSTYLE

    while True:
        if banner:
            print(str_in_color(banner, Color.GRAY))
        choices = [
            questionary.Choice(option["label"], value=str(option["choice"]))
            for option in payload.get("options", [])
        ]
        choices.append(questionary.Choice(_VIEW_OPTION_LABEL, value="v"))
        question = questionary.select(
            "Send the sanitized context to the cloud?",
            choices=choices,
            style=QSTYLE,
            qmark=QMARK,
        )
        question.application.erase_when_done = True
        choice = question.ask()
        if banner:
            _erase_printed(banner)
        if choice is None:  # the human quit the prompt
            return _gate_choice(payload, "reject")
        if choice in _VIEW_CHOICES:
            _show_sanitized(payload)
            continue
        if choice in (_gate_choice(payload, "retry"), "retry"):
            feedback = (
                questionary.text(
                    "What should the pipeline do differently? (Enter to skip)",
                    style=QSTYLE,
                    qmark=QMARK,
                ).ask()
                or ""
            ).strip()
            if feedback:
                return {"choice": choice, "feedback": feedback}
        return choice


def _stdin_gate(payload: dict) -> object:
    """Numbered fallback for terminals questionary cannot drive."""
    try:
        while True:
            print()
            _print_menu(payload)
            choice = input(str_brand("Gate > ", bold=True)).strip()
            if choice.lower() in _VIEW_CHOICES:
                _show_sanitized(payload)
                continue
            if choice.lower() in (_gate_choice(payload, "retry"), "retry"):
                feedback = input("Feedback (optional, Enter to skip) > ").strip()
                if feedback:
                    return {"choice": choice, "feedback": feedback}
            return choice
    except EOFError:
        raise RuntimeError(
            "The privacy gate needs an interactive terminal to ask for approval, "
            "but stdin is closed. Set 'interactive: false' in the privacy config."
        ) from None


def _menu_available() -> bool:
    """questionary needs a real terminal, and ships only with the `tui` extra."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    return importlib.util.find_spec("questionary") is not None


def make_terminal_approver(review_card: bool = True) -> Approver:
    """Build an approver that asks the human at the gate."""

    def approve(payload: dict) -> object:
        with paused_progress():
            error = payload.get("error")
            if error:
                print(str_in_color(error, Color.RED))
            elif review_card:
                _print_review(payload)
            headline = None if review_card else payload.get("headline")
            if _menu_available():
                return _menu_gate(payload, headline)
            if headline:
                print(str_in_color(headline, Color.GRAY))
            return _stdin_gate(payload)

    return approve


terminal_approver = make_terminal_approver()


def validate_privacy_config(config: PrivacyConfig) -> None:
    """Fail now, not mid-query, on a config the pipeline cannot run."""
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
    approver: Optional[Approver] = None,
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


def load_privacy_config(path: str) -> PrivacyConfig:
    """Load a privacy config from a YAML path and check it right away."""
    config = load_config(path, PrivacyConfig)
    validate_privacy_config(config)
    return config


def setup_privacy(
    config: Union[PrivacyConfig, str],
    *,
    interactive_ok: bool = True,
    review_card: bool = True,
) -> Tuple[Any, Optional[Approver], PrivacyConfig]:
    """Compile the graph from a config, or from the path to one."""
    from langgraph.checkpoint.memory import MemorySaver

    from .pipeline import build_privacy_pipeline

    if isinstance(config, PrivacyConfig):
        validate_privacy_config(config)
    else:
        config = load_privacy_config(config)
    approver: Optional[Approver] = None
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
