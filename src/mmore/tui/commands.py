"""Registry of mmore commands callable from the TUI.

Each entry mirrors a Click command in `mmore.cli` so the TUI is a thin wrapper:
the `run` callable is the same `run_*` function the CLI uses.
"""

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class CommandSpec:
    name: str
    description: str
    example_config: Optional[str]
    run: Callable[..., None]
    needs_input_data: bool = False
    config_globs: list[str] = field(default_factory=list)
    # Lazy importer returning the dataclass to validate YAML against.
    # Returns None if no validation is wired up for this stage.
    config_dataclass: Optional[Callable[[], Any]] = None
    # Extras the user has to `uv sync --extra ...` for this stage to import.
    # Used only to build a friendly install hint.
    required_extras: list[str] = field(default_factory=list)
    # Module names probed via `importlib.util.find_spec` to verify the extras
    # are actually installed. If any is missing, the stage is disabled in the
    # menu with an install hint.
    canary_imports: list[str] = field(default_factory=list)


def check_stage_available(spec: "CommandSpec") -> Optional[str]:
    """Return None if all canary imports resolve, else an install-hint string."""
    missing: list[str] = []
    for mod in spec.canary_imports:
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except (ImportError, ValueError):
            missing.append(mod)
    if not missing:
        return None
    extras = " ".join(f"--extra {e}" for e in spec.required_extras)
    return f"Missing: {', '.join(missing)}. Install with: uv sync {extras}".strip()


def _process(config_file: str, **_):
    from mmore.run_process import process

    process(config_file)


def _postprocess(config_file: str, input_data: str, **_):
    from mmore.run_postprocess import postprocess

    postprocess(config_file, input_data)


def _index(
    config_file: str,
    documents_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    **_,
):
    from mmore.run_index import index

    index(config_file, documents_path, collection_name)


def _rag(config_file: str, **_):
    from mmore.run_rag import rag

    rag(config_file)


def _ragcli(config_file: str, **_):
    from mmore.run_ragcli import RagCLI

    RagCLI(config_file).launch_cli()


def _websearch(config_file: str, **_):
    from mmore.run_websearch import run_websearch

    run_websearch(config_file)


def _colvision_process(config_file: str, **_):
    from mmore.colvision.run_process import run_process

    run_process(config_file)


def _colvision_index(config_file: str, **_):
    from mmore.colvision.run_index import index

    index(config_file)


def _colvision_retrieve(config_file: str, **_):
    from mmore.colvision.run_retriever import run_api

    run_api(config_file, "0.0.0.0", 8001)


# Lazy dataclass importers — keeps heavy deps out of TUI startup.
def _dc_process():
    from mmore.run_process import ProcessInference

    return ProcessInference


def _dc_postprocess():
    from mmore.process.post_processor.pipeline import PPPipelineConfig

    return PPPipelineConfig


def _dc_index():
    from mmore.run_index import IndexConfig

    return IndexConfig


def _dc_rag():
    from mmore.run_rag import RAGInferenceConfig

    return RAGInferenceConfig


REGISTRY: dict[str, CommandSpec] = {
    "process": CommandSpec(
        name="process",
        description="Crawl + extract documents into a JSONL",
        example_config="examples/process/config.yaml",
        run=_process,
        config_globs=[
            "examples/process/**/*.yaml",
            "examples/process/**/*.yml",
        ],
        config_dataclass=_dc_process,
        required_extras=["process", "cpu"],
        canary_imports=["torch", "marker", "transformers"],
    ),
    "postprocess": CommandSpec(
        name="postprocess",
        description="Chunk / clean processed documents",
        example_config="examples/postprocessor/config.yaml",
        run=_postprocess,
        needs_input_data=True,
        config_globs=[
            "examples/postprocessor/**/*.yaml",
            "examples/postprocessor/**/*.yml",
        ],
        config_dataclass=_dc_postprocess,
        required_extras=["process", "cpu"],
        canary_imports=["torch", "transformers"],
    ),
    "index": CommandSpec(
        name="index",
        description="Embed + store documents in Milvus",
        example_config="examples/index/config.yaml",
        run=_index,
        config_globs=[
            "examples/index/**/*.yaml",
            "examples/index/**/*.yml",
        ],
        config_dataclass=_dc_index,
        required_extras=["index", "cpu"],
        canary_imports=["pymilvus", "sentence_transformers", "torch"],
    ),
    "rag": CommandSpec(
        name="rag",
        description="Run a one-shot RAG pipeline",
        example_config="examples/rag/config.yaml",
        run=_rag,
        config_globs=[
            "examples/rag/**/*.yaml",
            "examples/rag/**/*.yml",
        ],
        config_dataclass=_dc_rag,
        required_extras=["rag", "cpu"],
        canary_imports=["langchain", "pymilvus", "torch"],
    ),
    "ragcli": CommandSpec(
        name="ragcli",
        description="Interactive RAG chat",
        example_config="examples/rag/config.yaml",
        run=_ragcli,
        config_globs=[
            "examples/rag/**/*.yaml",
            "examples/rag/**/*.yml",
        ],
        config_dataclass=_dc_rag,
        required_extras=["rag", "cpu"],
        canary_imports=["langchain", "pymilvus", "torch"],
    ),
    "websearch": CommandSpec(
        name="websearch",
        description="Web search (+ optional RAG)",
        example_config="examples/websearchRAG/config.yaml",
        run=_websearch,
        config_globs=[
            "examples/websearchRAG/**/*.yaml",
            "examples/websearchRAG/**/*.yml",
        ],
        required_extras=["websearch"],
        canary_imports=["ddgs"],
    ),
    "colvision-process": CommandSpec(
        name="colvision-process",
        description="Embed PDF pages with ColVision",
        example_config="examples/colvision/config_process.yml",
        run=_colvision_process,
        config_globs=["examples/colvision/**/*.yaml", "examples/colvision/**/*.yml"],
        required_extras=["colvision", "cpu"],
        canary_imports=["colpali_engine", "torch"],
    ),
    "colvision-index": CommandSpec(
        name="colvision-index",
        description="Store ColVision embeddings in Milvus",
        example_config="examples/colvision/config_index.yml",
        run=_colvision_index,
        config_globs=["examples/colvision/**/*.yaml", "examples/colvision/**/*.yml"],
        required_extras=["colvision", "cpu"],
        canary_imports=["colpali_engine", "pymilvus"],
    ),
    "colvision-retrieve": CommandSpec(
        name="colvision-retrieve",
        description="Run ColVision retriever API",
        example_config="examples/colvision/config_retrieval.yml",
        run=_colvision_retrieve,
        config_globs=["examples/colvision/**/*.yaml", "examples/colvision/**/*.yml"],
        required_extras=["colvision", "api", "cpu"],
        canary_imports=["colpali_engine", "pymilvus", "fastapi"],
    ),
}
