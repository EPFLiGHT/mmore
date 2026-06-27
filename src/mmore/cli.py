import os
from typing import Optional

import click

from mmore.profiler import enable_profiling_from_env

# To not have the fork-handler messages ("FD from fork parent still in poll list")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")


@click.group()
def main():
    """CLI for mmore commands."""
    enable_profiling_from_env()
    pass


@main.command()
@click.option(
    "--config-file", type=str, required=True, help="Dispatcher configuration file path."
)
def process(config_file: str):
    """Process documents from a directory.

    Args:
      config_file: Dispatcher configuration file path.

    Returns:

    """
    from .run_process import process as run_process

    run_process(config_file)


@main.command()
@click.option(
    "--config-file",
    type=str,
    required=True,
    help="Path to the config file for post-processing.",
)
@click.option(
    "--input-data",
    type=str,
    required=True,
    help="Path to the input JSONL file of documents.",
)
def postprocess(config_file: str, input_data: str):
    """Run the post-processors pipeline.

    Args:
      config_file: path to the config file for post-processing.
      input_data: path to the input JSONL file of documents.

    Returns:

    """
    from .run_postprocess import postprocess as run_postprocess

    run_postprocess(config_file, input_data)


@main.command()
@click.option(
    "--config-file",
    "-c",
    type=str,
    required=True,
    help="Path to the config file for indexing.",
)
@click.option(
    "--documents-path",
    "-f",
    type=str,
    required=False,
    help="Path to the JSONL file of the (post)processed documents.",
)
@click.option(
    "--collection-name",
    "-n",
    type=str,
    required=False,
    help="Name that will be used to refer to this collection of documents.",
)
def index(config_file: str, documents_path: str, collection_name: str):
    """Run the indexer.

    Args:
      config_file: path to the config file for indexing.
      documents_path: path to the JSONL file of the (post)processed documents.
      collection_name: name that will be used to refer to this collection of documents.

    Returns:

    """
    from .run_index import index as run_index

    run_index(config_file, documents_path, collection_name)


@main.command()
@click.option(
    "--config-file",
    "-c",
    type=str,
    required=True,
    help="Retriever configuration file path.",
)
@click.option(
    "--input-file",
    "-f",
    type=str,
    required=False,
    default=None,
    help="Path to the JSONL file of the input queries.",
)
@click.option(
    "--output-file",
    "-o",
    type=str,
    required=False,
    default=None,
    help="Path to which save the results of the retriever as a JSON.",
)
@click.option(
    "--host", type=str, default="0.0.0.0", help="Host on which the API should be run."
)
@click.option(
    "--port", type=int, default=8001, help="Port on which the API should be run."
)
def retrieve(
    config_file: str,
    input_file: Optional[str],
    output_file: Optional[str],
    host: str,
    port: int,
):
    """Retrieve documents for specified queries.

    Args:
      config_file: path to the config file for the retriever.
      input_file: path to the JSONL file of the input queries.
      output_file: path to which save the results of the retriever as a JSON.

    Returns:

    """
    from .run_retriever import retrieve as run_retrieve
    from .run_retriever import run_api

    if input_file:
        assert isinstance(output_file, str)
        run_retrieve(config_file, input_file, output_file)
    else:
        run_api(config_file, host, port)


@main.command()
@click.option(
    "--config-file",
    "-c",
    type=str,
    required=True,
    help="Retriever configuration file path.",
)
@click.option(
    "--host", type=str, default="0.0.0.0", help="Host on which the API should be run."
)
@click.option(
    "--port", type=int, default=8000, help="Port on which the API should be run."
)
def live_retrieval(config_file: str, host: str, port: int):
    """API for live indexing and retrieval of documents.

    Args:
      config_file: Path to the retriever configuration file.
      host: Host on which the API should be run.
      port: Port on which the API should be run.
    """
    from .run_live_retrieval import run

    run(config_file, host, port)


@main.command()
@click.option(
    "--config-file", type=str, required=True, help="Dispatcher configuration file path."
)
def rag(config_file: str):
    """Run the Retrieval-Augmented Generation (RAG) pipeline.

    Args:
      config_file: Dispatcher configuration file path.

    Returns:

    """
    from .run_rag import rag as run_rag

    run_rag(config_file)


@main.command()
@click.option(
    "--config-file",
    "-c",
    type=str,
    required=True,
    help="Retriever configuration file path.",
)
@click.option(
    "--host", type=str, default="0.0.0.0", help="Host on which the API should be run."
)
@click.option(
    "--port", type=int, default=8000, help="Port on which the API should be run."
)
def index_api(config_file, host, port):
    """Run the Index API.

    Args:
      config_file: Path to the retriever configuration file.
      host: Host on which the API should be run.
      port: Port on which the API should be run.

    Returns:

    """
    from .run_index_api import run_api

    run_api(config_file, host, port)


@main.command()
@click.option(
    "--config-file",
    type=str,
    required=True,
    help="Path to the Websearch configuration file (YAML).",
)
def websearch(config_file):
    """Run the Websearch (+ optional RAG) pipeline."""
    from .run_websearch import run_websearch

    # # Load your YAML configuration and pass it into the runner
    # with open(config_file, "r") as f:
    #     config_dict = yaml.safe_load(f)

    run_websearch(config_file)


@main.command(name="paper-discovery")
@click.option(
    "--config-file",
    type=str,
    required=True,
    help="Path to the Paper Discovery configuration file (YAML),",
)
def paper_discovery(config_file):
    """Run the Paper Discovery pipeline (fetch papers from academic repos)."""
    from .run_paper_discovery import run_paper_discovery

    run_paper_discovery(config_file)


@main.command()
@click.option(
    "--config-file", type=str, required=True, help="Configuration for the RAG CLI."
)
def ragcli(config_file: str):
    """Run the RAG CLI.

    Args:
      config_file: Configuration.

    Returns:

    """
    from .run_ragcli import RagCLI

    my_rag_cli = RagCLI(config_file)
    my_rag_cli.launch_cli()


@main.command()
def tui():
    """Launch the interactive Terminal UI."""
    try:
        from .tui import run
    except ModuleNotFoundError as e:
        if e.name in ("questionary", "rich", "prompt_toolkit"):
            click.echo("TUI dependency missing. Install with: uv sync --extra tui")
            raise SystemExit(1)
        raise
    run()


@main.group()
def colvision():
    """ColVision pipeline commands for PDF processing, indexing, and retrieval."""
    pass


@colvision.command(name="process")
@click.option(
    "--config-file",
    type=str,
    required=True,
    help="Path to the ColVision process configuration file.",
)
@click.option(
    "--model",
    "-m",
    type=str,
    required=False,
    default=None,
    help="Override the model_name from the config file (e.g. vidore/colqwen2.5-v0.2).",
)
def colvision_process(config_file: str, model: Optional[str]):
    """Process PDFs and generate page embeddings using a ColVision model.

    Args:
      config_file: Path to the ColVision process configuration file.
      model: Optional model name override.

    Returns:

    """
    from .colvision.run_process import run_process

    run_process(config_file, model_name_override=model)


@colvision.command(name="index")
@click.option(
    "--config-file",
    "-c",
    type=str,
    required=True,
    help="Path to the ColVision index configuration file.",
)
def colvision_index(config_file: str):
    """Index ColVision embeddings into a Milvus database.

    Args:
      config_file: Path to the ColVision index configuration file.

    Returns:

    """
    from .colvision.run_index import index as run_colvision_index

    run_colvision_index(config_file)


@colvision.command(name="retrieve")
@click.option(
    "--config-file",
    "-c",
    type=str,
    required=True,
    help="Path to the ColVision retriever configuration file.",
)
@click.option(
    "--model",
    "-m",
    type=str,
    required=False,
    default=None,
    help="Override the model_name from the config file (e.g. vidore/colqwen2.5-v0.2).",
)
@click.option(
    "--input-file",
    "-f",
    type=str,
    required=False,
    default=None,
    help="Path to the JSONL file of the input queries.",
)
@click.option(
    "--output-file",
    "-o",
    type=str,
    required=False,
    default=None,
    help="Path to which save the results of the retriever as a JSON.",
)
@click.option(
    "--host", type=str, default="0.0.0.0", help="Host on which the API should be run."
)
@click.option(
    "--port", type=int, default=8001, help="Port on which the API should be run."
)
def colvision_retrieve(
    config_file: str,
    model: Optional[str],
    input_file: Optional[str],
    output_file: Optional[str],
    host: str,
    port: int,
):
    """Retrieve documents using ColVision embeddings.

    Args:
      config_file: Path to the ColVision retriever configuration file.
      model: Optional model name override.
      input_file: Path to the JSONL file of the input queries.
      output_file: Path to which save the results of the retriever as a JSON.
      host: Host on which the API should be run.
      port: Port on which the API should be run.

    Returns:

    """
    from .colvision.run_retriever import retrieve as run_colvision_retrieve
    from .colvision.run_retriever import run_api as run_colvision_api

    if input_file:
        if output_file is None:
            raise ValueError(
                "Both --input-file and --output-file must be provided together."
            )
        run_colvision_retrieve(
            config_file, input_file, output_file, model_name_override=model
        )
    else:
        run_colvision_api(config_file, host, port, model_name_override=model)


if __name__ == "__main__":
    main()
