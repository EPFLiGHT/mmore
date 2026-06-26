import argparse
import time
from dataclasses import dataclass
from typing import Optional, Union

from dotenv import load_dotenv

from mmore.index.indexer import Indexer, IndexerConfig
from mmore.profiler import enable_profiling_from_env, profile_function
from mmore.type import MultimodalSample
from mmore.utils import load_config
from mmore.ux import (
    model_loading_seconds,
    quiet_noisy_libs,
    setup_logging,
    step_intro,
    step_summary,
)

INDEX_NAME = "Index"
INDEX_EMOJI = "📇"
logger = setup_logging(INDEX_NAME, INDEX_EMOJI)

load_dotenv()


@dataclass
class IndexConfig:
    indexer: IndexerConfig
    collection_name: str
    documents_path: str


@profile_function()
def index(
    config_file: Union[IndexConfig, str],
    documents_path: Optional[str] = None,
    collection_name: Optional[str] = None,
):
    """Index files for specified documents."""
    quiet_noisy_libs()
    # Load the config file
    config: IndexConfig = load_config(config_file, IndexConfig)
    if collection_name is None:
        collection_name = config.collection_name
    if documents_path is None:
        documents_path = config.documents_path

    documents = MultimodalSample.from_jsonl(documents_path)

    step_intro(
        INDEX_NAME,
        INDEX_EMOJI,
        "Make your documents searchable and store them",
        [f"{len(documents)} documents", f"collection: {collection_name}"],
    )

    start = time.time()
    loading_start = model_loading_seconds()
    Indexer.from_documents(
        config=config.indexer, documents=documents, collection_name=collection_name
    )
    elapsed = time.time() - start - (model_loading_seconds() - loading_start)
    step_summary(
        INDEX_NAME,
        INDEX_EMOJI,
        elapsed,
        {
            "embedding": config.indexer.dense_model.model_name,
            "throughput": f"{len(documents) / elapsed:.1f} docs/s" if elapsed else "-",
        },
    )


if __name__ == "__main__":
    enable_profiling_from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", required=True, help="Path to the index configuration file."
    )
    parser.add_argument(
        "--documents-path", "-f", required=False, help="Path to the JSONL data."
    )
    parser.add_argument(
        "--collection-name",
        "-n",
        required=False,
        help="Name of the collection to index.",
    )
    args = parser.parse_args()

    index(args.config_file, args.documents_path, args.collection_name)
