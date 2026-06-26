"""mmore TUI entry point."""

from __future__ import annotations

import threading

import questionary
from rich.panel import Panel
from rich.text import Text

from mmore.tui.commands import REGISTRY, check_stage_available
from mmore.tui.config_builder import (
    build_full_pipeline_wizard,
    pick_or_build_config,
)
from mmore.tui.exceptions import UserCancelledError
from mmore.tui.paths import cwd_default
from mmore.tui.pipeline import run_full_pipeline, run_pipeline_with_configs
from mmore.tui.theme import (
    ACCENT,
    ACCENT2,
    MUTED,
    QMARK,
    QSTYLE,
    console,
    run_step,
    section,
    select,
    show_banner,
)

_PIPELINE_STAGES = ("process", "postprocess", "index")


def _warm_pipeline_dataclasses() -> None:
    """Pre-load process/postprocess/index dataclasses in a daemon thread.

    Called when entering the wizard or full-pipeline flows, where several YAML
    validations happen back-to-back. The import cost then overlaps with the
    wizard's own prompts. Daemon = no impact on exit. Stages whose canary
    imports are missing are skipped so partial installs don't crash the warm-up.
    """

    def _warm() -> None:
        for stage in _PIPELINE_STAGES:
            spec = REGISTRY[stage]
            if check_stage_available(spec) is not None or spec.config_dataclass is None:
                continue
            try:
                spec.config_dataclass()
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=_warm, daemon=True).start()


def _show_missing_extras(spec_name: str, hint: str) -> None:
    console.print(
        Panel(
            Text.assemble(
                (f"Stage `{spec_name}` can't run.\n\n", "bold"),
                (hint, "yellow"),
            ),
            title="[bold yellow]missing dependencies[/]",
            border_style="yellow",
            padding=(1, 2),
        )
    )


def _missing_extras_notice() -> Panel | None:
    """One-line-per-install-command notice — kept compact so the banner stays visible."""
    install_to_stages: dict[str, list[str]] = {}
    for name, spec in REGISTRY.items():
        hint = check_stage_available(spec)
        if hint and "Install with: " in hint:
            cmd = hint.split("Install with: ", 1)[1].strip()
            install_to_stages.setdefault(cmd, []).append(name)

    if not install_to_stages:
        return None

    body = Text()
    for i, (cmd, stages) in enumerate(install_to_stages.items()):
        if i > 0:
            body.append("\n")
        body.append(", ".join(stages), style="bold white")
        body.append("  →  ", style="yellow")
        body.append(cmd, style="cyan")

    return Panel(
        body,
        title="[bold yellow]⚠  missing extras[/]",
        border_style="yellow",
        padding=(0, 1),
    )


def _disabled_label(label: str) -> str:
    """Prefix a menu label so its disabled state is immediately readable."""
    return f"⚠  {label}"


def _is_colvision(spec) -> bool:
    return spec.name.startswith("colvision-")


def _select_and_run(specs, display_name, prompt: str) -> None:
    """Show a command picker for `specs` and run the chosen one. `display_name`
    maps a spec to its menu label text."""
    choices = []
    enabled_count = 0
    displays = {spec.name: display_name(spec) for spec in specs}
    width = max(len(d) for d in displays.values())
    for spec in specs:
        hint = check_stage_available(spec)
        label = f"{displays[spec.name]:<{width}} — {spec.description}"
        if hint:
            choices.append(
                questionary.Choice(
                    _disabled_label(label), value=spec.name, disabled="missing extras"
                )
            )
        else:
            choices.append(questionary.Choice(label, value=spec.name))
            enabled_count += 1

    # questionary crashes ("InquirerControl has no attribute 'pointed_at'") when
    # every choice is disabled because it can't pick an initial pointer. Bail
    # out with a clear notice instead.
    if enabled_count == 0:
        notice = _missing_extras_notice()
        if notice is not None:
            console.print(notice)
        return

    name = select(prompt, choices, answer_labels=displays)
    if name is not None:
        _run_spec(name)


def _run_spec(name: str) -> None:
    spec = REGISTRY[name]
    # Defensive re-check in case the user typed past the disabled state.
    hint = check_stage_available(spec)
    if hint:
        _show_missing_extras(spec.name, hint)
        return
    config_file = pick_or_build_config(spec)
    kwargs = {"config_file": config_file}
    if spec.needs_input_data:
        input_data = questionary.text(
            "Input JSONL path",
            default=cwd_default("examples/process/outputs/merged/merged_results.jsonl"),
            style=QSTYLE,
            qmark=QMARK,
        ).ask()
        if input_data is None:
            return
        kwargs["input_data"] = input_data

    interactive = name in {"ragcli", "rag", "colvision-retrieve"}
    if interactive:
        spec.run(**kwargs)
    else:
        run_step(spec.description, spec.run, **kwargs)
    console.print()


def _run_single_command() -> None:
    # ColVision is a separate sub menu
    specs = [s for s in REGISTRY.values() if not _is_colvision(s)]
    _select_and_run(specs, lambda s: s.name, "Pick a command")


def _run_colvision_menu() -> None:
    specs = [s for s in REGISTRY.values() if _is_colvision(s)]
    _select_and_run(
        specs,
        lambda s: s.name.removeprefix("colvision-"),
        "Pick a ColVision command",
    )


def _chat_only() -> None:
    config_file = pick_or_build_config(REGISTRY["ragcli"])
    REGISTRY["ragcli"].run(config_file=config_file)


def _run_full_wizard() -> None:
    _warm_pipeline_dataclasses()
    paths = build_full_pipeline_wizard()
    console.print()
    console.print(
        section(
            "Wizard complete",
            Text(
                "process:     " + paths["process"] + "\n"
                "postprocess: " + paths["postprocess"] + "\n"
                "index:       " + paths["index"],
                style=MUTED,
            ),
            style=ACCENT2,
        )
    )
    if questionary.confirm(
        "Run the pipeline now with these configs?",
        default=True,
        style=QSTYLE,
        qmark=QMARK,
    ).ask():
        run_pipeline_with_configs(
            paths["process"], paths["postprocess"], paths["index"]
        )


def _pipeline_hint() -> str | None:
    """Return a combined hint if any of process/postprocess/index is missing."""
    hints = [
        check_stage_available(REGISTRY[s]) for s in ("process", "postprocess", "index")
    ]
    hints = [h for h in hints if h]
    return " | ".join(hints) if hints else None


def _colvision_hint() -> str | None:
    hints = [check_stage_available(s) for s in REGISTRY.values() if _is_colvision(s)]
    if hints and all(h is not None for h in hints):
        return next(h for h in hints if h)
    return None


def _main_menu() -> str | None:
    notice = _missing_extras_notice()
    if notice is not None:
        console.print(notice)

    pipeline_hint = _pipeline_hint()
    chat_hint = check_stage_available(REGISTRY["ragcli"])
    colvision_hint = _colvision_hint()
    # The wizard validates each generated YAML against the stage's dataclass,
    # which transitively imports torch / transformers / etc. — so it needs the
    # same extras as the full pipeline. Reuse `_pipeline_hint()` to stay aligned.
    wizard_hint = _pipeline_hint()

    pipeline_label = "🚀 Run full pipeline  (process → postprocess → index)"
    wizard_label = "🧙 Build a full pipeline config (guided wizard)"
    chat_label = "💬 Chat with indexed documents"
    colvision_label = "🖼  ColVision  (process → index → retrieve)"

    pipeline_choice = questionary.Choice(
        _disabled_label(pipeline_label) if pipeline_hint else pipeline_label,
        value="pipeline",
        disabled="missing extras" if pipeline_hint else None,
    )
    wizard_choice = questionary.Choice(
        _disabled_label(wizard_label) if wizard_hint else wizard_label,
        value="wizard",
        disabled="missing extras" if wizard_hint else None,
    )
    chat_choice = questionary.Choice(
        _disabled_label(chat_label) if chat_hint else chat_label,
        value="chat",
        disabled="missing extras" if chat_hint else None,
    )
    colvision_choice = questionary.Choice(
        _disabled_label(colvision_label) if colvision_hint else colvision_label,
        value="colvision",
        disabled="missing extras" if colvision_hint else None,
    )

    return select(
        "What do you want to do?",
        choices=[
            questionary.Separator("Default pipeline (text-based) ──"),
            questionary.Choice("🟢 Run a single command", value="single"),
            pipeline_choice,
            wizard_choice,
            chat_choice,
            questionary.Separator(" "),
            questionary.Separator("Alternative pipeline (vision-based) ──"),
            colvision_choice,
            questionary.Separator(),
            questionary.Choice("🔧 Setup (install dependencies)", value="setup"),
            questionary.Choice("✕  Quit", value="quit"),
        ],
    )


def run() -> None:
    console.clear()
    show_banner("interactive launcher")
    while True:
        # Ctrl-C at the main menu itself quits; inside any sub-flow it
        # cancels and returns here.
        try:
            mode = _main_menu()
        except KeyboardInterrupt:
            console.print(f"\n[{ACCENT}]bye![/]")
            return
        if mode in (None, "quit"):
            console.print(f"[{ACCENT}]bye![/]")
            return

        try:
            if mode == "single":
                _run_single_command()
            elif mode == "pipeline":
                _warm_pipeline_dataclasses()
                run_full_pipeline()
            elif mode == "wizard":
                _run_full_wizard()
            elif mode == "chat":
                _chat_only()
            elif mode == "colvision":
                _run_colvision_menu()
            elif mode == "setup":
                from mmore.tui.setup import run_setup_wizard

                run_setup_wizard()
        except (UserCancelledError, KeyboardInterrupt):
            console.print("[white]cancelled — back to menu.[/]")
            continue
        except Exception as e:  # noqa: BLE001
            console.print(f"[bold red]error:[/] {e}")
            try:
                cont = questionary.confirm(
                    "Continue?", default=True, style=QSTYLE
                ).ask()
            except KeyboardInterrupt:
                return
            if not cont:
                return
