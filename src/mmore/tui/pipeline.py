"""Chain process -> postprocess -> index from the TUI."""

from __future__ import annotations

import questionary
from rich.table import Table
from rich.text import Text

from mmore.tui.commands import REGISTRY
from mmore.tui.config_builder import pick_or_build_config, pick_privacy_config
from mmore.tui.inspector import inspect_jsonl
from mmore.tui.theme import (
    ACCENT,
    ACCENT2,
    MUTED,
    console,
    run_step,
    section,
    step_header,
)


def _process_output_jsonl(config_path: str) -> str:
    """Resolve the JSONL path the `process` step writes to.

    Goes through `mmore.utils.load_config` so env-var expansion ($ROOT_OUT_DIR,
    etc.) matches what the underlying command sees.
    """
    from mmore.run_process import ProcessInference, merged_results_path
    from mmore.utils import load_config

    cfg: ProcessInference = load_config(config_path, ProcessInference)
    return merged_results_path(cfg.dispatcher_config.output_path)


def _postprocess_output_jsonl(config_path: str) -> str:
    """Resolve the JSONL path `postprocess` writes to.

    Mirrors `PPPipeline`'s use of `mmore.process.utils.jsonl_path`: if the
    configured `output_path` is a directory, the pipeline writes to
    `<dir>/final.jsonl`; if it already ends in `.jsonl`, it's used as-is.
    """
    from mmore.process.post_processor.pipeline import PPPipelineConfig
    from mmore.process.utils import jsonl_path
    from mmore.utils import load_config

    cfg: PPPipelineConfig = load_config(config_path, PPPipelineConfig)
    return jsonl_path(cfg.output.output_path)


def _summary_table(rows: list[tuple[str, str, float]]) -> Table:
    table = Table(
        title="[bold]Pipeline summary[/bold]",
        title_style=ACCENT2,
        border_style=ACCENT,
        header_style=f"bold {ACCENT}",
        show_lines=False,
    )
    table.add_column("Step", style="bold")
    table.add_column("Output", style=MUTED)
    table.add_column("Duration", justify="right")
    total = 0.0
    for name, out, dur in rows:
        table.add_row(name, out, f"{dur:.1f}s")
        total += dur
    table.add_section()
    table.add_row("[bold]Total[/bold]", "", f"[bold]{total:.1f}s[/bold]")
    return table


def run_pipeline_with_configs(process_cfg: str, pp_cfg: str, index_cfg: str) -> None:
    """Execute the three stages given already-built YAML paths."""
    console.print()
    console.print(
        section(
            "Full pipeline",
            Text("process → postprocess → index → (optional) chat", style=ACCENT),
            style=ACCENT2,
        )
    )

    rows: list[tuple[str, str, float]] = []

    step_header(1, 3, "process")
    elapsed = run_step(
        "Crawling + extracting documents",
        REGISTRY["process"].run,
        config_file=process_cfg,
    )
    process_jsonl = _process_output_jsonl(process_cfg)
    rows.append(("process", process_jsonl, elapsed))
    inspect_jsonl(process_jsonl)

    step_header(2, 3, "postprocess")
    elapsed = run_step(
        "Chunking + cleaning",
        REGISTRY["postprocess"].run,
        config_file=pp_cfg,
        input_data=process_jsonl,
    )
    pp_jsonl = _postprocess_output_jsonl(pp_cfg)
    rows.append(("postprocess", pp_jsonl, elapsed))
    inspect_jsonl(pp_jsonl)

    step_header(3, 3, "index")
    elapsed = run_step(
        "Embedding + indexing into Milvus",
        REGISTRY["index"].run,
        config_file=index_cfg,
        documents_path=pp_jsonl,
    )
    rows.append(("index", "(vector DB)", elapsed))

    console.print()
    console.print(_summary_table(rows))
    console.print()

    if questionary.confirm("Open the RAG chat now?", default=True).ask():
        rag_cfg = pick_or_build_config(REGISTRY["ragcli"])
        REGISTRY["ragcli"].run(
            config_file=rag_cfg, privacy_config_file=pick_privacy_config()
        )


def run_full_pipeline() -> None:
    console.print()
    console.print(
        section(
            "Full pipeline",
            Text("process → postprocess → index → (optional) chat", style=ACCENT),
            style=ACCENT2,
        )
    )

    rows: list[tuple[str, str, float]] = []

    step_header(1, 3, "process")
    process_cfg = pick_or_build_config(REGISTRY["process"])
    elapsed = run_step(
        "Crawling + extracting documents",
        REGISTRY["process"].run,
        config_file=process_cfg,
    )
    process_jsonl = _process_output_jsonl(process_cfg)
    rows.append(("process", process_jsonl, elapsed))
    inspect_jsonl(process_jsonl)

    step_header(2, 3, "postprocess")
    pp_cfg = pick_or_build_config(REGISTRY["postprocess"])
    elapsed = run_step(
        "Chunking + cleaning",
        REGISTRY["postprocess"].run,
        config_file=pp_cfg,
        input_data=process_jsonl,
    )
    pp_jsonl = _postprocess_output_jsonl(pp_cfg)
    rows.append(("postprocess", pp_jsonl, elapsed))
    inspect_jsonl(pp_jsonl)

    step_header(3, 3, "index")
    index_cfg = pick_or_build_config(REGISTRY["index"], documents_path=pp_jsonl)
    elapsed = run_step(
        "Embedding + indexing into Milvus",
        REGISTRY["index"].run,
        config_file=index_cfg,
        documents_path=pp_jsonl,
    )
    rows.append(("index", "(vector DB)", elapsed))

    console.print()
    console.print(_summary_table(rows))
    console.print()

    if questionary.confirm("Open the RAG chat now?", default=True).ask():
        rag_cfg = pick_or_build_config(REGISTRY["ragcli"])
        REGISTRY["ragcli"].run(
            config_file=rag_cfg, privacy_config_file=pick_privacy_config()
        )
