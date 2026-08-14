"""Shared UX helpers for all mmore pipelines: standard logging setup, colored
output, a spinner, and a progress bar."""

import io
import itertools
import logging
import os
import random
import re
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterable, Iterator, Optional, Sized

from rich.color import Color as RichColor
from rich.color import ColorSystem
from rich.console import COLOR_SYSTEMS, Console, Group, RenderableType
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.spinner import Spinner as RichSpinner
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

DATEFMT = "%Y-%m-%d %H:%M:%S"

# Default step identity for the `process` pipeline
PROCESS_NAME = "Process"
PROCESS_EMOJI = "🚀"

# Set MMORE_VERBOSE=1 to unquiet everything
VERBOSE_ENV = "MMORE_VERBOSE"

# Strip the leading decoration (emoji, ★, ▶ ...) up to the first word/path char
DECORATION = re.compile(r"^[^\w/~.]+\s*")

# Third-party loggers names
_NOISY_LIBS = [
    "surya",
    "marker",
    "transformers",
    "moviepy",
    "datasets",
    "sentence_transformers",
    "PIL",
    "httpx",
    "urllib3",
    "primp",
    "reqwest",
    "hyper_util",
    "h2",
    "hickory_net",
    "hickory_resolver",
    "rustls",
    "cookie_store",
    "presidio-analyzer",
    "spacy",
    "gliner",
    "langgraph",
    "langchain",
    "dspy",
    "LiteLLM",
    "litellm",
    "openai",
    "httpcore",
]


def is_verbose() -> bool:
    """True when MMORE_VERBOSE is set."""
    return bool(os.environ.get(VERBOSE_ENV))


# ------------------------------ Logging setup ------------------------------ #


class _LateBoundStderr(io.TextIOBase):
    """Looks sys.stderr up on every write, so logs follow a live display's proxy
    instead of scrolling the terminal behind its back."""

    def write(self, text: str) -> int:
        return sys.stderr.write(text)

    def flush(self) -> None:
        sys.stderr.flush()


def setup_logging(name: str, emoji: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging with the standard mmore header `[name emoji time] msg`
    and return a logger for the step. Call once from a pipeline entry point."""
    logging.basicConfig(
        format=f"[{name} {emoji} %(asctime)s] %(message)s",
        datefmt=DATEFMT,
        level=logging.DEBUG if is_verbose() else level,
        handlers=[logging.StreamHandler(_LateBoundStderr())],
        force=True,
    )
    return logging.getLogger(name)


def _hide_download_bars() -> None:
    """Hide Hugging Face download bars."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from huggingface_hub.utils.tqdm import disable_progress_bars
    except ImportError:
        return
    disable_progress_bars()


def quiet_noisy_libs(hide_info: bool = False) -> None:
    """Silence noisy third-party libraries and their progress bars so output
    stays readable. With hide_info=True, also hide mmore's own INFO logs."""
    if is_verbose():
        return
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    _hide_download_bars()
    warnings.filterwarnings("ignore")
    for name in _NOISY_LIBS:
        logging.getLogger(name).setLevel(logging.ERROR)
    if hide_info:
        logging.disable(logging.INFO)
    try:
        from transformers.utils import logging as hf_logging
    except ImportError:
        return
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()


def init_worker(name: str = PROCESS_NAME, emoji: str = PROCESS_EMOJI) -> None:
    """Reconfigure logging inside a worker process so its logging config
    is the same as the parent one."""
    setup_logging(name, emoji)
    quiet_noisy_libs()


# ----------------------------- Colored output ------------------------------ #


class Color:
    MMORE = "#f7cb46"  # TODO: change once Malo finishes the new logo
    ACCENT = "#5fd7ff"
    ACCENT2 = "#ff79c6"
    RED = "#ff5555"
    GREEN = "#50fa7b"
    YELLOW = "#f1fa8c"
    BLUE = "#6cb6ff"
    ORANGE = "#ffaf00"
    GRAY = "#888888"


# Text styles
_RESET = "\033[0m"
_BOLD = "\033[1m"


@lru_cache(maxsize=None)
def _ansi_for(color: str) -> str:
    """ANSI translation for a `#rrggbb` hex (cached to not recompute each time)."""
    system = COLOR_SYSTEMS.get(_console().color_system or "", ColorSystem.STANDARD)
    codes = RichColor.parse(color).downgrade(system).get_ansi_codes(foreground=True)
    return "\033[" + ";".join(codes) + "m"


def str_in_color(to_print: str | int, color: str, bold: bool = False) -> str:
    style = _ansi_for(color)
    if bold:
        style = _BOLD + style
    return f"{style}{to_print}{_RESET}"


def print_in_color(to_print: str | int, color: str, bold: bool = False) -> None:
    print(str_in_color(to_print, color, bold))


def str_green(text: str | int, bold: bool = False) -> str:
    return str_in_color(text, Color.GREEN, bold=bold)


def str_brand(text: str | int, bold: bool = False) -> str:
    return str_in_color(text, Color.MMORE, bold=bold)


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


# --------------------------------- Spinner --------------------------------- #

SPINNER_WORDS = [
    "Thinking",
    "Pondering",
    "Discombobulating",
    "Cooking",
    "Brewing",
    "Ruminating",
    "Rummaging",
    "Noodling",
]

# Seconds a spinner sticks with a word before picking the next one
_WORD_SECONDS = 3.0


# Only the innermost active spinner draws, so a nested download spinner can take
# over the shared line and hand it back on exit
_SPINNER_STACK: "list[Spinner]" = []


class _SpinnerWords:
    """A spinner's caption: a word, and the seconds it has been running."""

    def __init__(self, label: Optional[str] = None) -> None:
        self._label = label
        self._word = label or random.choice(SPINNER_WORDS)
        self._start = self._word_time = time.monotonic()

    def caption(self) -> str:
        now = time.monotonic()
        if self._label is None and now - self._word_time > _WORD_SECONDS:
            self._word, self._word_time = random.choice(SPINNER_WORDS), now
        return f"{self._word}... ({int(now - self._start)}s)"


class _WordSpinner(RichSpinner):
    """The same caption, animated by a live display instead of its own thread."""

    def __init__(self) -> None:
        super().__init__("dots", style=Color.MMORE)
        self.words = _SpinnerWords()

    def render(self, time: float) -> RenderableType:
        self.text = Text(self.words.caption(), style=Color.MMORE)
        return super().render(time)


class Spinner:
    """Animated status line shown while work happens in the calling thread."""

    def __init__(self, label: Optional[str] = None) -> None:
        self._label = label
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._saved_status: Optional[list[str]] = None
        self._saved_words: Optional[_SpinnerWords] = None

    def __enter__(self) -> "Spinner":
        self._stop.clear()
        _SPINNER_STACK.append(self)
        if _SPIN is not None:
            if self._label is not None:
                self._saved_words = _SPIN.words
                _SPIN.words = _SpinnerWords(self._label)
        elif _LIVE is not None:
            self._saved_status = list(_STATUS)
            label = self._label or random.choice(SPINNER_WORDS)
            line = f"  [{Color.MMORE}]{escape(label)}...[/]"
            _set_status([*self._saved_status[:-1], line])
        elif sys.stdout.isatty():
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self in _SPINNER_STACK:
            _SPINNER_STACK.remove(self)
        if self._saved_words is not None and _SPIN is not None:
            _SPIN.words = self._saved_words
            self._saved_words = None
        if self._saved_status is not None:
            _set_status(self._saved_status)
            self._saved_status = None
        if self._thread is not None:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def _spin(self) -> None:
        frames = itertools.cycle("|/-\\")
        words = _SpinnerWords(self._label)
        while not self._stop.is_set():
            # Stay quiet as long as a nested spinner is running
            if not _SPINNER_STACK or _SPINNER_STACK[-1] is not self:
                time.sleep(0.1)
                continue
            status = f"{next(frames)} {words.caption()}"
            sys.stdout.write(f"\r\033[K{str_in_color(status, Color.MMORE)}")
            sys.stdout.flush()
            time.sleep(0.1)


_DOWNLOAD_SECONDS = 0.0


def model_loading_seconds() -> float:
    """Seconds spent in loading_model()."""
    return _DOWNLOAD_SECONDS


@contextmanager
def loading_model(name: str) -> "Iterator[None]":
    """Spinner naming the model while it loads (and downloads on first run)."""
    global _DOWNLOAD_SECONDS
    start = time.monotonic()
    try:
        if is_verbose():
            yield
        else:
            _hide_download_bars()
            with Spinner(f"Loading {name}"):
                yield
    finally:
        _DOWNLOAD_SECONDS += time.monotonic() - start


# ------------------------------ Branded cards ------------------------------ #

_CONSOLE: Optional[Console] = None


def _console() -> Console:
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = Console(
            theme=Theme(
                {
                    "bar.complete": Color.MMORE,
                    "bar.finished": Color.MMORE,
                    "bar.pulse": Color.MMORE,
                    "bar.back": "grey30",
                    "progress.percentage": f"bold {Color.MMORE}",
                    "progress.download": Color.GRAY,
                    "progress.elapsed": "grey46",
                }
            ),
            highlight=False,
        )
    return _CONSOLE


# Config fields stashed by step_intro for step_summary to render
_INTRO_FIELDS: dict[str, list[str]] = {}


def step_intro(
    step: str, emoji: str, about: str, fields: Optional[list[str]] = None
) -> None:
    """Print the one-line opener."""
    setup_logging(step, emoji)
    _INTRO_FIELDS[step] = list(fields) if fields else []

    line = f"[bold {Color.MMORE}]▸ {escape(step)} {emoji}[/]  {escape(about)}"
    if fields:
        line += f" [dim]·[/] [{Color.MMORE}]{escape(fields[0])}[/]"
    _console().print(line)


def card(title: str, rows: "dict[str, object]") -> None:
    """Branded panel with a label/value table, for in-run reports."""
    table = Table.grid(padding=(0, 3))
    table.add_column(style="dim")
    table.add_column()
    for label, value in rows.items():
        table.add_row(escape(str(label)), escape(str(value)))

    _console().print(
        Panel(
            table,
            title=f"[bold]mmore[/] ▸ {escape(title)}",
            title_align="left",
            border_style=Color.MMORE,
            expand=False,
        )
    )


def step_summary(
    step: str, emoji: str, elapsed: float, stats: "dict[str, object]"
) -> None:
    """Print the closing card: the run's config (left) and result stats (right)."""
    config = _INTRO_FIELDS.pop(step, None)
    items = [(escape(str(k)), escape(str(v))) for k, v in stats.items()]
    table = Table.grid(padding=(0, 3))

    if config:
        table.add_column(style="dim")  # config
        table.add_column(style="dim")  # stat label
        table.add_column()  # stat value
        for i in range(max(len(config), len(items))):
            left = escape(config[i]) if i < len(config) else ""
            label, value = items[i] if i < len(items) else ("", "")
            table.add_row(left, label, value)
    else:
        table.add_column(style="dim")
        table.add_column()
        for label, value in items:
            table.add_row(label, value)

    _console().print(
        Panel(
            table,
            title=f"[bold]mmore[/] ▸ {step} {emoji} · done in {elapsed:.1f}s",
            title_align="left",
            border_style=Color.MMORE,
            expand=False,
        )
    )


# ------------------------------ Progress bar ------------------------------- #

_PROGRESS: Optional[Progress] = None
_PROGRESS_REFS = 0
_LIVE: Optional[Live] = None
_STATUS: list[str] = []
_SPIN: Optional[_WordSpinner] = None


def _live_group() -> Group:
    lines: list = [Text.from_markup(line) for line in _STATUS]
    if _PROGRESS is not None:
        lines.append(_PROGRESS)
    elif _SPIN is not None:
        lines.append(_SPIN)
    return Group(*lines)


def _ensure_live() -> None:
    global _LIVE
    if _LIVE is None:
        _LIVE = Live(_live_group(), console=_console(), refresh_per_second=12)
        _LIVE.start()


def _erase_live() -> None:
    global _LIVE
    if _LIVE is None:
        return
    _LIVE.transient = True
    _LIVE.stop()
    _LIVE = None


def _ensure_progress() -> Progress:
    global _PROGRESS
    if _PROGRESS is None:
        _PROGRESS = Progress(
            SpinnerColumn(style=Color.MMORE),
            TextColumn("{task.description}"),
            BarColumn(complete_style=Color.MMORE, finished_style=Color.MMORE),
            MofNCompleteColumn(),
            TextColumn("{task.fields[unit]}", style="dim", markup=False),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[postfix]}", style="dim", markup=False),
            console=_console(),
        )
        _ensure_live()
    return _PROGRESS


def _set_status(lines: "Iterable[str]") -> None:
    """Replace the lines shown just above the bar (empty to drop them)."""
    global _STATUS
    _STATUS = list(lines)
    if _LIVE is not None:
        _LIVE.update(_live_group())


_PAUSED_SECONDS = 0.0


def paused_seconds() -> float:
    """Seconds the bar spent paused, i.e. spent waiting on the user."""
    return _PAUSED_SECONDS


@contextmanager
def paused_progress() -> "Iterator[None]":
    """When the user interacts during the privacy HITL, it pauses the progress."""
    global _PAUSED_SECONDS, _LIVE
    if _LIVE is None:
        yield
        return
    _erase_live()
    start = time.monotonic()
    try:
        yield
    finally:
        _PAUSED_SECONDS += time.monotonic() - start
        _ensure_live()


def _release_progress() -> None:
    global _PROGRESS, _PROGRESS_REFS, _STATUS, _LIVE
    _PROGRESS_REFS = max(0, _PROGRESS_REFS - 1)
    if _PROGRESS_REFS or _PROGRESS is None or _LIVE is None:
        return
    _STATUS = []
    _LIVE.update(_live_group())
    _LIVE.stop()
    _PROGRESS, _LIVE = None, None


class _Status:
    """Live status lines above a spinner: the same surface as `_Bar`, for a run
    with nothing to count."""

    def __init__(self) -> None:
        global _SPIN
        _SPIN = _WordSpinner()
        _ensure_live()
        self._closed = False

    def set_status(self, *lines: str) -> None:
        _set_status(lines)

    def set_unit(self, unit: str | None) -> None:
        """Ignored: only a bar has a column for the step name."""

    def print_above(self, text: str) -> None:
        _console().print(text)

    def close(self) -> None:
        global _SPIN, _STATUS
        if self._closed:
            return
        self._closed = True
        _erase_live()
        _SPIN, _STATUS = None, []

    def __enter__(self) -> "_Status":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def live_status() -> _Status:
    """A spinner with status lines above it. Use as a context manager."""
    return _Status()


class _Bar:
    """A task in the shared rich Progress."""

    def __init__(
        self,
        iterable: Iterable | None,
        total: int | None,
        desc: str,
        unit: str,
    ) -> None:
        global _PROGRESS_REFS
        self._iterable = iterable
        if total is None and isinstance(iterable, Sized):
            total = len(iterable)
        self._prog = _ensure_progress()
        _PROGRESS_REFS += 1
        self._task = self._prog.add_task(desc, total=total, postfix="", unit=unit)
        self._closed = False

    def __iter__(self) -> "Iterator":
        assert self._iterable is not None
        try:
            for item in self._iterable:
                yield item
                self._prog.advance(self._task)
        finally:
            self.close()

    def set_postfix_str(self, text: str) -> None:
        self._prog.update(self._task, postfix=str(text))

    def set_unit(self, unit: str | None) -> None:
        self._prog.update(self._task, unit=unit or "")

    def set_status(self, *lines: str) -> None:
        """Rich-markup lines redrawn in place above the bar (none to clear)."""
        _set_status(lines)

    def print_above(self, text: str) -> None:
        self._prog.console.print(text)

    def update(self, n: int = 1) -> None:
        self._prog.advance(self._task, n)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Drop the current-file label once bar is finished
        self._prog.update(self._task, postfix="")
        _release_progress()

    def __enter__(self) -> "_Bar":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def progress(
    iterable: Optional[Iterable] = None,
    *,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "doc",
    **kwargs: object,
) -> _Bar:
    """Branded progress bar for per-item pipeline loops. Shows the brand spinner,
    bar, count, percentage, elapsed and ETA, plus an optional live postfix.

    Iterating closes the bar automatically. For the manual `total=`/`update()`
    pattern, use it as a context manager so the bar closes:

        ```with progress(total=n, desc="Indexing") as bar:
            bar.update(len(batch))
        ```
    """
    return _Bar(iterable, total, desc, unit)
