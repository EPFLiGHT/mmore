"""Themed questionary prompts.

All helpers raise UserCancelledError on Esc or Ctrl-C.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional

import questionary
from rich.markup import escape

from mmore.tui.exceptions import UserCancelledError
from mmore.tui.theme import ACCENT, QMARK, QSTYLE, console
from mmore.ux import DECORATION, Color


def ask(prompt_obj: questionary.Question) -> Any:
    """Call .ask() and translate Ctrl-C / Esc into UserCancelledError.

    questionary raises KeyboardInterrupt on Ctrl-C and returns None on Esc.
    Both should land the caller back at its menu, not exit the TUI.
    """
    try:
        answer = prompt_obj.ask()
    except KeyboardInterrupt as e:
        raise UserCancelledError("cancelled") from e
    if answer is None:
        raise UserCancelledError("cancelled")
    return answer


def _choice_title(value: Any, choices: List[Any]) -> str:
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
    text = DECORATION.sub("", text)
    text = re.sub(r"\s*\(recommended\)$", "", text)
    home = str(Path.home())
    if text.startswith(home):
        text = "~" + text[len(home) :]
    return text.strip()


def select(
    question: str,
    choices: List[Any],
    default: Optional[str] = None,
    answer_labels: Optional[dict[str, str]] = None,
) -> str:
    """Themed `questionary.select` with a uniform answer echo."""
    value = ask(
        questionary.select(
            question,
            choices=choices,
            default=default,
            style=QSTYLE,
            qmark=QMARK,
            erase_when_done=True,
        )
    )
    if answer_labels and value in answer_labels:
        answer = answer_labels[value]
    else:
        answer = _clean_answer(_choice_title(value, choices))
    console.print(
        f"[{ACCENT}]{QMARK}[/] [bold]{escape(question)}[/] "
        f"[bold {Color.MMORE}]{escape(answer)}[/]"
    )
    return value


def prompt(question: str, default: str = "") -> str:
    return ask(questionary.text(question, default=default, style=QSTYLE, qmark=QMARK))


def confirm(question: str, default: bool = False) -> bool:
    return ask(
        questionary.confirm(question, default=default, style=QSTYLE, qmark=QMARK)
    )


def prompt_int(question: str, default: int) -> int:
    try:
        return int(prompt(question, str(default)))
    except ValueError:
        return default


def prompt_float(question: str, default: float) -> float:
    try:
        return float(prompt(question, str(default)))
    except ValueError:
        return default
