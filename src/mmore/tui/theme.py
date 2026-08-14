"""Shared visuals: banner, palette, panel helpers."""

from __future__ import annotations

import time
from typing import Any, Callable

from questionary import Style
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from mmore.ux import Color

console = Console(highlight=False)

# Role styles derived from the shared palette (mmore.ux.Color)
ACCENT = Color.ACCENT
ACCENT2 = Color.ACCENT2
MUTED = Color.GRAY
OK = f"bold {Color.GREEN}"
WARN = str(Color.YELLOW)
ERR = f"bold {Color.RED}"

QSTYLE = Style(
    [
        ("qmark", f"fg:{ACCENT} bold"),
        ("question", "bold"),
        ("answer", f"fg:{Color.MMORE} bold"),
        ("pointer", f"fg:{ACCENT} bold"),
        ("highlighted", f"fg:{ACCENT} bold"),
        ("selected", f"fg:{Color.MMORE}"),
        ("instruction", f"fg:{MUTED} italic"),
        ("disabled", f"fg:{Color.ORANGE} italic"),
    ]
)
QMARK = "▸"


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
        out.append(mid, style=f"bold {Color.MMORE}")
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
