"""Generate YAML config files via guided prompts.

Templates here mirror the example configs under `examples/`. The user is
asked only for the fields most likely to change between runs; everything else
falls back to the example defaults. The resulting dict is dumped to a YAML
file under `./tui-configs/`.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import questionary
import yaml
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

from mmore.tui.commands import CommandSpec
from mmore.tui.exceptions import UserCancelledError
from mmore.tui.paths import cwd_default, repo_root, resolve_example
from mmore.tui.theme import ACCENT, ACCENT2, QMARK, QSTYLE, console, section, select


def _ask(prompt_obj: Any) -> Any:
    """Call .ask() and translate Ctrl-C / Esc into UserCancelledError.

    questionary raises KeyboardInterrupt on Ctrl-C and returns None on Esc.
    Both should land us back at the main menu, not exit the TUI.
    """
    try:
        answer = prompt_obj.ask()
    except KeyboardInterrupt as e:
        raise UserCancelledError("cancelled") from e
    if answer is None:
        raise UserCancelledError("cancelled")
    return answer


CONFIG_DIR = Path("./tui-configs")


def _select(
    question: str,
    choices: Any,
    default: Any = None,
    answer_labels: Optional[dict[Any, str]] = None,
) -> Any:
    """`theme.select` with this module's cancel semantics (Esc/Ctrl-C -> back)."""
    try:
        value = select(question, choices, answer_labels=answer_labels, default=default)
    except KeyboardInterrupt as e:
        raise UserCancelledError("cancelled") from e
    if value is None:
        raise UserCancelledError("cancelled")
    return value


def _prompt(question: str, default: str = "") -> str:
    return _ask(questionary.text(question, default=default, style=QSTYLE, qmark=QMARK))


def _confirm(question: str, default: bool = False) -> bool:
    return _ask(
        questionary.confirm(question, default=default, style=QSTYLE, qmark=QMARK)
    )


def _prompt_int(question: str, default: int) -> int:
    try:
        return int(_prompt(question, str(default)))
    except ValueError:
        return default


def _prompt_float(question: str, default: float) -> float:
    try:
        return float(_prompt(question, str(default)))
    except ValueError:
        return default


def _save(name: str, data: dict[str, Any]) -> str:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{name}-{time.time_ns()}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(path)


def _preview_config(path: str) -> None:
    """Display a YAML file with syntax highlighting."""
    content = Path(path).read_text()
    console.print(
        Panel(
            Syntax(content, "yaml", theme="monokai", line_numbers=True),
            title=f"[bold]{path}[/bold]",
            border_style=ACCENT,
            padding=(1, 2),
        )
    )


def _edit_config(path: str) -> None:
    """Open a config file in $EDITOR (falls back to vi).

    Supports editors with flags like ``EDITOR="code -w"`` via shlex.split.
    """
    editor = os.environ.get("EDITOR", "vi")
    subprocess.call([*shlex.split(editor), path])


def _post_validation_menu(path: str, spec: CommandSpec) -> str:
    """After validation, let the user preview, edit, or run the config.

    Returns the (potentially re-validated) path.
    """
    while True:
        action = _select(
            "What next?",
            choices=[
                questionary.Choice("▶  Run with this config", value="run"),
                questionary.Choice("👁  Preview config", value="preview"),
                questionary.Choice("✎  Edit in $EDITOR", value="edit"),
            ],
            default="run",
        )
        if action == "run":
            return path
        if action == "preview":
            _preview_config(path)
            continue
        if action == "edit":
            _edit_config(path)
            err = _validate_with_spinner(path, spec)
            if err:
                _show_error_panel(path, err)
            continue
    return path  # unreachable but keeps mypy happy


def build_process_config() -> str:
    data_path = _prompt(
        "Data path (folder with documents to process)",
        cwd_default("data"),
    )
    output_path = _prompt(
        "Output path (where merged_results.jsonl will be written)",
        cwd_default("outputs/process"),
    )
    use_fast = _confirm("Use fast (lower-quality) processors?", default=False)
    distributed = _confirm("Use distributed processing (Dask)?", default=False)
    extract_images = _confirm("Extract images from documents?", default=True)

    cfg = {
        "data_path": data_path,
        "google_drive_ids": [],
        "previous_results": None,
        "dispatcher_config": {
            "output_path": output_path,
            "use_fast_processors": use_fast,
            "distributed": distributed,
            "extract_images": extract_images,
            "scheduler_file": None,
            "process_batch_sizes": [
                {"URLProcessor": 40},
                {"DOCXProcessor": 100},
                {"PDFProcessor": 4000},
                {"MediaProcessor": 40},
                {"SpreadsheetProcessor": 100},
                {"TXTProcessor": 100},
                {"PPTXProcessor": 100},
                {"MarkdownProcessor": 100},
                {"EMLProcessor": 100},
                {"HTMLProcessor": 100},
            ],
            "processor_config": {
                "MediaProcessor": [
                    {"normal_model": "openai/whisper-large-v3-turbo"},
                    {"fast_model": "openai/whisper-tiny"},
                    {"type": "automatic-speech-recognition"},
                    {"sample_rate": 10},
                    {"batch_size": 4},
                ],
                "PDFProcessor": [
                    {"PDFTEXT_CPU_WORKERS": 0},
                    {"DETECTOR_BATCH_SIZE": 1},
                    {"DETECTOR_POSTPROCESSING_CPU_WORKERS": 0},
                    {"RECOGNITION_BATCH_SIZE": 1},
                    {"OCR_PARALLEL_WORKERS": 0},
                    {"TEXIFY_BATCH_SIZE": 1},
                    {"LAYOUT_BATCH_SIZE": 1},
                    {"ORDER_BATCH_SIZE": 1},
                    {"TABLE_REC_BATCH_SIZE": 1},
                ],
            },
        },
    }
    return _save("process", cfg)


def build_postprocess_config() -> str:
    strategy = _select(
        "Chunking strategy",
        choices=["sentence", "token", "word", "semantic"],
        default="sentence",
    )
    table_handling = _select(
        "Table handling",
        choices=["single_row", "multi_rows", "keep_whole", "none"],
        default="single_row",
    )
    output_path = _prompt(
        "Output JSONL path",
        cwd_default("outputs/postprocess/results.jsonl"),
    )

    cfg = {
        "previous_results": None,
        "pp_modules": [
            {
                "type": "chunker",
                "args": {
                    "chunking_strategy": strategy,
                    "table_handling": table_handling,
                },
            },
        ],
        "output": {"output_path": output_path, "save_each_step": True},
    }
    return _save("postprocess", cfg)


def build_index_config(documents_path: Optional[str] = None) -> str:
    dense = _prompt("Dense embedding model", "sentence-transformers/all-MiniLM-L6-v2")
    sparse = _prompt("Sparse embedding model", "splade")
    db_uri = _prompt(
        "DB URI (Milvus Lite file or server URL)", cwd_default("proc_demo.db")
    )
    db_name = _prompt("DB name", "my_db")
    collection = _prompt("Collection name", "my_docs")
    docs = documents_path or _prompt(
        "Documents JSONL path",
        cwd_default("outputs/postprocess/results.jsonl"),
    )
    cfg = {
        "indexer": {
            "dense_model": {"model_name": dense, "is_multimodal": False},
            "sparse_model": {"model_name": sparse, "is_multimodal": False},
            "db": {"uri": db_uri, "name": db_name},
        },
        "collection_name": collection,
        "documents_path": docs,
    }
    return _save("index", cfg)


def build_rag_config() -> str:
    """Wizard for `rag` / `retrieve` / `ragcli` configs."""
    llm_name = _prompt("LLM name", "OpenMeditron/meditron3-8b")
    max_new_tokens = _prompt_int("Max new tokens", 1200)

    db_uri = _prompt(
        "DB URI (Milvus Lite file or server URL)", cwd_default("proc_demo.db")
    )
    db_name = _prompt("DB name", "my_db")
    collection = _prompt("Collection name", "my_docs")
    k = _prompt_int("Number of docs to retrieve (k)", 5)
    hybrid = _prompt_float("Hybrid search weight (0.0 dense — 1.0 sparse)", 0.5)
    use_web = _confirm("Augment retrieval with web search?", default=False)
    reranker = _prompt("Reranker model (blank to skip)", "BAAI/bge-reranker-base")

    mode = _select(
        "Run mode",
        choices=["local", "api"],
        default="local",
    )

    cfg: dict[str, Any] = {
        "rag": {
            "llm": {"llm_name": llm_name, "max_new_tokens": max_new_tokens},
            "retriever": {
                "db": {"uri": db_uri, "name": db_name},
                "hybrid_search_weight": hybrid,
                "k": k,
                "collection_name": collection,
                "use_web": use_web,
                "reranker_model_name": reranker or None,
            },
            "system_prompt": (
                "Use the following context to answer the questions.\n\n"
                "Context:\n{context}"
            ),
        },
        "mode": mode,
    }
    if mode == "local":
        input_file = _prompt(
            "Queries JSONL path", resolve_example("examples/rag/queries.jsonl")
        )
        output_file = _prompt(
            "Output JSON path", cwd_default("outputs/rag/output.json")
        )
        cfg["mode_args"] = {"input_file": input_file, "output_file": output_file}
    else:
        port = _prompt_int("API port", 8000)
        cfg["mode_args"] = {
            "endpoint": "/rag",
            "host": "0.0.0.0",
            "port": port,
        }
    return _save("rag", cfg)


def build_websearch_config() -> str:
    """Wizard for `websearch` configs."""
    use_rag = _confirm("Combine web search with RAG?", default=True)
    rag_path = ""
    if use_rag:
        rag_path = _prompt(
            "Path to a RAG config YAML",
            resolve_example("examples/rag/config.yaml"),
        )
    llm_name = _prompt("LLM name", "OpenMeditron/meditron3-8b")
    max_new_tokens = _prompt_int("Max new tokens", 1200)
    input_queries = _prompt(
        "Input queries JSONL", resolve_example("examples/rag/queries.jsonl")
    )
    output_file = _prompt(
        "Output JSON path",
        cwd_default("outputs/websearch/enhanced_results.json"),
    )
    n_subqueries = _prompt_int("Number of sub-queries per question", 2)
    max_searches = _prompt_int("Max searches per query", 5)
    provider = _select(
        "Search provider",
        choices=["duckduckgo"],
        default="duckduckgo",
    )

    cfg: dict[str, Any] = {
        "websearch": {
            "use_rag": use_rag,
            "rag_config_path": rag_path,
            "use_summary": True,
            "n_subqueries": n_subqueries,
            "input_queries": input_queries,
            "output_file": output_file,
            "n_loops": 2,
            "max_searches": max_searches,
            "search_provider": provider,
            "max_retries": 3,
            "max_context_tokens": 2048,
            "fast_tokenizer": False,
            "mode": "local",
            "llm_config": {
                "llm_name": llm_name,
                "max_new_tokens": max_new_tokens,
            },
        }
    }
    return _save("websearch", cfg)


BUILDERS = {
    "process": build_process_config,
    "postprocess": build_postprocess_config,
    "index": build_index_config,
    "rag": build_rag_config,
    "ragcli": build_rag_config,
    "websearch": build_websearch_config,
}


# Static list of processor class names — kept in sync with
# src/mmore/process/processors/*.py. Used by the full-pipeline wizard so the
# user can pick a subset rather than always shipping all 10.
_ALL_PROCESSORS: list[tuple[str, int]] = [
    ("PDFProcessor", 4000),
    ("DOCXProcessor", 100),
    ("PPTXProcessor", 100),
    ("MarkdownProcessor", 100),
    ("HTMLProcessor", 100),
    ("TXTProcessor", 100),
    ("EMLProcessor", 100),
    ("SpreadsheetProcessor", 100),
    ("MediaProcessor", 40),
    ("URLProcessor", 40),
]

_PROCESSOR_DEFAULT_CONFIG: dict[str, list[dict[str, Any]]] = {
    "MediaProcessor": [
        {"normal_model": "openai/whisper-large-v3-turbo"},
        {"fast_model": "openai/whisper-tiny"},
        {"type": "automatic-speech-recognition"},
        {"sample_rate": 10},
        {"batch_size": 4},
    ],
    "PDFProcessor": [
        {"PDFTEXT_CPU_WORKERS": 0},
        {"DETECTOR_BATCH_SIZE": 1},
        {"DETECTOR_POSTPROCESSING_CPU_WORKERS": 0},
        {"RECOGNITION_BATCH_SIZE": 1},
        {"OCR_PARALLEL_WORKERS": 0},
        {"TEXIFY_BATCH_SIZE": 1},
        {"LAYOUT_BATCH_SIZE": 1},
        {"ORDER_BATCH_SIZE": 1},
        {"TABLE_REC_BATCH_SIZE": 1},
    ],
}


def build_process_config_wizard() -> str:
    """Richer process-config builder that lets the user pick processors."""
    data_path = _prompt(
        "Data path (folder with documents to process)", cwd_default("data")
    )
    output_path = _prompt(
        "Output path (where merged_results.jsonl will be written)",
        cwd_default("outputs/process"),
    )
    use_fast = _confirm("Use fast (lower-quality) processors?", default=False)
    distributed = _confirm("Use distributed processing (Dask)?", default=False)
    extract_images = _confirm("Extract images from documents?", default=True)

    names = [n for n, _ in _ALL_PROCESSORS]
    selected = _ask(
        questionary.checkbox(
            "Select processors to enable",
            choices=[questionary.Choice(n, value=n, checked=True) for n in names],
            style=QSTYLE,
            qmark=QMARK,
        )
    )
    if not selected:
        selected = names  # empty would mean a no-op pipeline; fall back to all

    customize = _confirm("Customize batch sizes?", default=False)
    sizes: list[dict[str, int]] = []
    for name, default in _ALL_PROCESSORS:
        if name not in selected:
            continue
        value = _prompt_int(f"Batch size for {name}", default) if customize else default
        sizes.append({name: value})

    processor_config = {
        name: cfg for name, cfg in _PROCESSOR_DEFAULT_CONFIG.items() if name in selected
    }

    # Incremental resume: detect previous results
    from mmore.run_process import merged_results_path

    previous_results = None
    prev_path = merged_results_path(output_path)
    if os.path.exists(prev_path) and _confirm(
        f"Previous results found at {prev_path}. Resume (skip unchanged files)?",
        default=True,
    ):
        previous_results = prev_path

    cfg = {
        "data_path": data_path,
        "google_drive_ids": [],
        "previous_results": previous_results,
        "dispatcher_config": {
            "output_path": output_path,
            "use_fast_processors": use_fast,
            "distributed": distributed,
            "extract_images": extract_images,
            "scheduler_file": None,
            "process_batch_sizes": sizes,
            "processor_config": processor_config,
        },
    }
    return _save("process", cfg)


def _postprocessor_choices() -> list[str]:
    """Enumerate every post-processor `type` string the loader accepts.

    The wizard is reachable without the `process` extra installed (it only
    writes YAML), so we fall back to the core set if the extra modules are
    missing instead of crashing mid-wizard with an ImportError.
    """
    base = ["chunker", "ner", "translator", "metafuse"]
    try:
        from mmore.process.post_processor.filter import FILTER_TYPES
        from mmore.process.post_processor.tagger import TAGGER_TYPES
    except ImportError:
        return base
    return [*base, *TAGGER_TYPES, *FILTER_TYPES]


def _ask_module_args(pp_type: str) -> dict[str, Any]:
    if pp_type == "chunker":
        strategy = _select(
            "Chunking strategy",
            choices=["sentence", "token", "word", "semantic"],
            default="sentence",
        )
        table_handling = _select(
            "Table handling",
            choices=["single_row", "multi_rows", "keep_whole", "none"],
            default="single_row",
        )
        return {
            "chunking_strategy": strategy,
            "table_handling": table_handling,
        }
    if pp_type in {"ner", "translator", "metafuse"}:
        if _confirm(f"Provide extra args for `{pp_type}` as YAML?", default=False):
            raw = _prompt("YAML args (single line, e.g. {key: value})", "{}")
            try:
                parsed = yaml.safe_load(raw) or {}
                if isinstance(parsed, dict):
                    return parsed
            except yaml.YAMLError:
                pass
        return {}
    return {}


def build_postprocess_config_wizard() -> str:
    """Build a postprocess config with an arbitrary list of pp_modules."""
    available = _postprocessor_choices()
    modules: list[dict[str, Any]] = []
    while True:
        if modules:
            console.print(
                f"  [dim]current modules:[/] {', '.join(m['type'] for m in modules)}"
            )
        pp_type = _select(
            "Add a post-processor module" if not modules else "Add another module",
            choices=[*available, questionary.Separator(), "(done)"],
        )
        if pp_type == "(done)":
            break
        args = _ask_module_args(pp_type)
        modules.append({"type": pp_type, "args": args})

    output_path = _prompt(
        "Output JSONL path",
        cwd_default("outputs/postprocess/results.jsonl"),
    )

    # Incremental resume: detect previous results
    previous_results = None
    # Resolve the actual JSONL path (dir → dir/final.jsonl, .jsonl → as-is)
    if output_path.endswith(".jsonl"):
        pp_prev_path = output_path
    else:
        pp_prev_path = os.path.join(output_path, "final.jsonl")
    if os.path.exists(pp_prev_path) and _confirm(
        f"Previous results found at {pp_prev_path}. Resume (skip unchanged)?",
        default=True,
    ):
        previous_results = pp_prev_path

    cfg = {
        "previous_results": previous_results,
        "pp_modules": modules,
        "output": {"output_path": output_path, "save_each_step": True},
    }
    return _save("postprocess", cfg)


def build_index_config_wizard(documents_path: Optional[str] = None) -> str:
    dense = _prompt("Dense embedding model", "sentence-transformers/all-MiniLM-L6-v2")
    sparse = _prompt("Sparse embedding model", "splade")
    multimodal = _confirm("Multimodal embeddings?", default=False)
    db_uri = _prompt(
        "DB URI (Milvus Lite file or server URL)", cwd_default("proc_demo.db")
    )
    db_name = _prompt("DB name", "my_db")
    collection = _prompt("Collection name", "my_docs")
    docs = documents_path or _prompt(
        "Documents JSONL path",
        cwd_default("outputs/postprocess/results.jsonl"),
    )
    cfg = {
        "indexer": {
            "dense_model": {"model_name": dense, "is_multimodal": multimodal},
            "sparse_model": {"model_name": sparse, "is_multimodal": multimodal},
            "db": {"uri": db_uri, "name": db_name},
        },
        "collection_name": collection,
        "documents_path": docs,
    }
    return _save("index", cfg)


def build_full_pipeline_wizard() -> dict[str, str]:
    """Build process + postprocess + index configs in one flow.

    Wires the postprocess output JSONL into the index config's documents_path
    so the three files form a coherent pipeline. Validates each YAML and
    re-prompts on failure (the per-stage builders run again on retry).
    """
    from mmore.tui.commands import REGISTRY
    from mmore.tui.pipeline import _postprocess_output_jsonl

    console.print(section("Pipeline wizard", Text("step 1/3 — process", style=ACCENT2)))
    while True:
        process_path = build_process_config_wizard()
        err = _validate_with_spinner(process_path, REGISTRY["process"])
        if err is None:
            break
        _show_error_panel(process_path, err)
        if not _confirm("Retry the process step?", default=True):
            raise UserCancelledError("cancelled")

    console.print(
        section("Pipeline wizard", Text("step 2/3 — postprocess", style=ACCENT2))
    )
    while True:
        pp_path = build_postprocess_config_wizard()
        err = _validate_with_spinner(pp_path, REGISTRY["postprocess"])
        if err is None:
            break
        _show_error_panel(pp_path, err)
        if not _confirm("Retry the postprocess step?", default=True):
            raise UserCancelledError("cancelled")

    try:
        docs_jsonl = _postprocess_output_jsonl(pp_path)
    except Exception:  # noqa: BLE001
        docs_jsonl = None

    console.print(section("Pipeline wizard", Text("step 3/3 — index", style=ACCENT2)))
    while True:
        index_path = build_index_config_wizard(documents_path=docs_jsonl)
        err = _validate_with_spinner(index_path, REGISTRY["index"])
        if err is None:
            break
        _show_error_panel(index_path, err)
        if not _confirm("Retry the index step?", default=True):
            raise UserCancelledError("cancelled")

    return {"process": process_path, "postprocess": pp_path, "index": index_path}


def find_yaml_configs(spec: CommandSpec) -> list[str]:
    """Find candidate YAML configs scoped to this stage.

    Globs are evaluated against the resolved repo root (looked up by walking
    up from CWD), so the TUI works from any working directory. Generated
    configs in `./tui-configs/` (CWD-relative) are always included so users
    keep access to configs they just built.
    """
    root = repo_root() or Path.cwd()
    matches: list[str] = []
    for pattern in spec.config_globs:
        for p in root.glob(pattern):
            matches.append(str(p))
    generated = Path.cwd() / "tui-configs"
    if generated.exists():
        for p in sorted(generated.glob(f"{spec.name}-*.yaml")):
            matches.append(str(p))

    seen: set[str] = set()
    out: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _validate_yaml(path: str, spec: CommandSpec) -> Optional[str]:
    """Return None on success, an error message string on failure."""
    if spec.config_dataclass is None:
        return None
    try:
        from mmore.utils import load_config

        dataclass_cls = spec.config_dataclass()
        load_config(path, dataclass_cls)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def _validate_with_spinner(path: str, spec: CommandSpec) -> Optional[str]:
    """Same as _validate_yaml but shows a spinner — config dataclass imports
    can take several seconds (heavy transitive imports), making the TUI look
    frozen otherwise."""
    spinner = Spinner(
        "dots", text=Text(f"  Validating {spec.name} config…", style="cyan")
    )
    result: dict[str, Optional[str]] = {}
    with Live(spinner, console=console, refresh_per_second=12, transient=True):
        result["err"] = _validate_yaml(path, spec)
    return result["err"]


def _show_error_panel(path: str, err: str) -> None:
    console.print(
        Panel(
            Text.assemble(
                (f"{path}\n\n", "bold"),
                (err, "red"),
            ),
            title="[bold red]invalid config[/]",
            border_style="red",
            padding=(1, 2),
        )
    )


def _ranked_choices(spec: CommandSpec, candidates: list[str]) -> list[Any]:
    """Put `spec.example_config` first as ★ recommended; rest under a separator."""
    choices: list[Any] = []
    rec_resolved: Optional[str] = None
    if spec.example_config:
        rec_resolved = resolve_example(spec.example_config)
    rest = list(candidates)
    if rec_resolved and rec_resolved in rest:
        choices.append(
            questionary.Choice(f"★ {rec_resolved}  (recommended)", value=rec_resolved)
        )
        rest.remove(rec_resolved)
    elif rec_resolved and Path(rec_resolved).exists():
        choices.append(
            questionary.Choice(f"★ {rec_resolved}  (recommended)", value=rec_resolved)
        )
    if rest:
        if choices:
            choices.append(questionary.Separator("── other configs ──"))
        for c in rest:
            choices.append(questionary.Choice(c, value=c))
    return choices


def pick_or_build_config(
    spec: CommandSpec, documents_path: Optional[str] = None
) -> str:
    """Ask the user to either pick an existing YAML or generate one.

    Validates the chosen YAML against the stage's dataclass and re-prompts
    on failure rather than letting the run blow up later.
    """
    while True:
        choice = _select(
            f"Config for `{spec.name}`?",
            choices=[
                questionary.Choice("📂 Pick existing YAML", value="pick"),
                questionary.Choice("✨ Generate new YAML (guided)", value="build"),
                questionary.Choice("✎  Edit an existing YAML in $EDITOR", value="edit"),
                questionary.Choice("⌨  Type a path manually", value="manual"),
            ],
        )

        path: Optional[str] = None

        if choice in ("pick", "edit"):
            candidates = find_yaml_configs(spec)
            ranked = _ranked_choices(spec, candidates)
            if not ranked:
                questionary.print(
                    f"No YAML configs found for `{spec.name}`, "
                    "falling back to manual entry.",
                    style="fg:yellow",
                )
                choice = "manual"
            else:
                path = _select(
                    f"Select a config for `{spec.name}`",
                    choices=ranked,
                )
                if choice == "edit" and path is not None:
                    _edit_config(path)

        if choice == "manual":
            manual = _prompt("Path to YAML config")
            manual = os.path.expandvars(os.path.expanduser(manual))
            if not os.path.exists(manual):
                _show_error_panel(manual, "file not found")
                continue
            path = manual

        if choice == "build":
            builder = BUILDERS.get(spec.name)
            if builder is None:
                questionary.print(
                    f"No guided builder for `{spec.name}` — pick an existing YAML.",
                    style="fg:yellow",
                )
                continue
            if spec.name == "index":
                path = builder(documents_path=documents_path)  # type: ignore[call-arg]
            else:
                path = builder()

        if path is None:
            raise UserCancelledError("no config selected")
        err = _validate_with_spinner(path, spec)
        if err is None:
            return _post_validation_menu(path, spec)
        _show_error_panel(path, err)
        if not _confirm("Try a different config?", default=True):
            raise UserCancelledError("cancelled")
