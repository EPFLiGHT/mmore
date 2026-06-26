"""Shared visuals: banner, palette, panel helpers."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import questionary
from questionary import Style
from rich.align import Align
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

console = Console()

# Palette
BRAND = "#f7cb46"
ACCENT = "bright_cyan"
ACCENT2 = "magenta"
MUTED = "grey58"
OK = "bold green"
WARN = "yellow"
ERR = "bold red"

QSTYLE = Style(
    [
        ("qmark", "fg:#5fd7ff bold"),
        ("question", "bold"),
        ("answer", f"fg:{BRAND} bold"),
        ("pointer", "fg:#5fd7ff bold"),
        ("highlighted", "fg:#5fd7ff bold"),
        ("selected", f"fg:{BRAND}"),
        ("instruction", "fg:#808080 italic"),
        ("disabled", "fg:#ffaf00 italic"),
    ]
)
QMARK = "▸"

# Strip the leading decoration (emoji, ★, ▶ ...) up to the first word/path char
_DECORATION = re.compile(r"^[^\w/~.]+\s*")


def _choice_title(value: Any, choices: Sequence[Any]) -> str:
    """The display title of the chosen value, joining formatted-text titles."""
    for c in choices:
        if isinstance(c, questionary.Choice):
            if c.value == value:
                title = c.title
                if isinstance(title, str):
                    return title
                if title is None:
                    return str(value)
                return "".join(tok[1] for tok in title)
        elif c == value:
            return str(c)
    return str(value)


def _clean_answer(title: str) -> str:
    """Reduce a menu label to a compact text."""
    text = re.sub(r"\s{2,}", " ", title.strip())
    text = _DECORATION.sub("", text)
    text = re.sub(r"\s*\(recommended\)$", "", text)
    home = str(Path.home())
    if text.startswith(home):
        text = "~" + text[len(home) :]
    return text.strip()


def select(
    question: str,
    choices: Sequence[Any],
    answer_labels: dict[Any, str] | None = None,
    **kwargs: Any,
) -> Any:
    """Themed `questionary.select` with a uniform answer echo."""
    value = questionary.select(
        question,
        choices=choices,
        style=QSTYLE,
        qmark=QMARK,
        erase_when_done=True,
        **kwargs,
    ).ask()
    if value is not None:
        if answer_labels and value in answer_labels:
            answer = answer_labels[value]
        else:
            answer = _clean_answer(_choice_title(value, choices))
        console.print(
            f"[{ACCENT}]{QMARK}[/] [bold]{escape(question)}[/] "
            f"[bold {BRAND}]{escape(answer)}[/]"
        )
    return value


BANNER = r"""

 ███╗   ███╗███╗   ███╗ ██████╗ ██████╗ ███████╗
 ████╗ ████║████╗ ████║██╔═══██╗██╔══██╗██╔════╝
 ██╔████╔██║██╔████╔██║██║   ██║██████╔╝█████╗
 ██║╚██╔╝██║██║╚██╔╝██║██║   ██║██╔══██╗██╔══╝
 ██║ ╚═╝ ██║██║ ╚═╝ ██║╚██████╔╝██║  ██║███████╗
 ╚═╝     ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
"""


def _mmore_logo(text: str) -> Text:
    """Color the banner like the mmore GitHub logo.

    Strategy, per character:
    - The second `M` (columns 12:23 of every row) is rendered in the brand color.
    - Elsewhere: outline characters (`╔╗╚╝═║╔╝╗`, etc.) are white and the
      filled `█` blocks are black, giving the letters a hollow look.
    """
    outline_chars = set("╔╗╚╝═║╠╣╦╩╬╔╝╗┌┐└┘─│")
    out = Text()
    for line in text.splitlines():
        if not line.strip():
            out.append(line + "\n")
            continue
        left = line[:12]
        mid = line[12:23]
        right = line[23:]

        def _emit(segment: str) -> None:
            for ch in segment:
                if ch == "█":
                    # explicit hex — terminal "black" often renders as dark grey
                    out.append(ch, style="#000000")
                elif ch in outline_chars:
                    out.append(ch, style="bold #ffffff")
                else:
                    out.append(ch)

        _emit(left)
        out.append(mid, style=f"bold {BRAND}")
        _emit(right)
        out.append("\n")
    return out


def show_banner(subtitle: str = "interactive launcher") -> None:
    body = Group(
        Align.center(_mmore_logo(BANNER)),
        Align.center(Text(subtitle, style=f"italic {MUTED}")),
    )
    console.print(
        Panel(
            body,
            border_style=ACCENT,
            padding=(0, 2),
        )
    )


def section(title: str, body: str | Text, style: str = ACCENT) -> Panel:
    return Panel(
        body if isinstance(body, Text) else Text(body),
        title=f"[bold]{title}[/bold]",
        border_style=style,
        padding=(1, 2),
    )


def run_step(label: str, fn: Callable[..., Any], **kwargs: Any) -> float:
    """Call fn(**kwargs) and return its clock duration."""
    start = time.time()
    fn(**kwargs)
    return time.time() - start


def step_header(idx: int, total: int, name: str) -> None:
    bar = "─" * 4
    console.print()
    console.print(
        f"[{ACCENT}]{bar}[/] [bold]Step {idx}/{total}[/bold] "
        f"[{ACCENT2}]{name}[/] [{ACCENT}]{bar}[/]"
    )
