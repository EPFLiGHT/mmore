import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Type, TypeVar, Union, cast

import yaml
from dacite import from_dict

if TYPE_CHECKING:
    from .index.indexer import Indexer
    from .rag.retriever import Retriever, RetrieverConfig
    from .type import MultimodalSample

T = TypeVar("T")


def expand_env_vars(obj):
    if isinstance(obj, dict):
        return {key: expand_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        return os.path.expandvars(obj)
    else:
        return obj


def load_config(yaml_dict_or_path: Union[str, Dict, T], config_class: Type[T]) -> T:
    if isinstance(yaml_dict_or_path, config_class):
        return yaml_dict_or_path

    if isinstance(yaml_dict_or_path, str):
        with open(yaml_dict_or_path, "r") as file:
            data = yaml.safe_load(file)
    else:
        data = yaml_dict_or_path

    # we want to support $ROOT_IN_DIR, $ROOT_OUT_DIR
    data = expand_env_vars(data)

    return from_dict(config_class, cast(Dict, data))


# Custom Dumper to preserve \n and avoid wrapping
class LiteralStringDumper(yaml.SafeDumper):
    pass


def str_presenter(dumper, data):
    if "\n" in data or "\r" in data:
        # Force double-quoted string with literal escapes
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


LiteralStringDumper.add_representer(str, str_presenter)


# Cache indexers in memory
indexers = {}
retrievers = {}


def create_new_indexer(
    collection_name: str, uri: str, db_name: str, device: Optional[str] = None
) -> "Indexer":
    """Create a new indexer with default configuration"""

    from .index.indexer import DBConfig, Indexer, IndexerConfig
    from .rag.model.dense.base import DenseModelConfig
    from .rag.model.sparse.base import SparseModelConfig

    try:
        # Default model configurations
        dense_config = DenseModelConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2", is_multimodal=False
        )

        sparse_config = SparseModelConfig(model_name="splade", is_multimodal=False)

        db_config = DBConfig(uri=uri, name=db_name)

        # Create indexer config
        config = IndexerConfig(
            dense_model=dense_config, sparse_model=sparse_config, db=db_config
        )

        # Create an empty list of documents for initialization
        empty_docs = []

        # Create indexer from documents (this will create the collection)
        indexer = Indexer.from_documents(
            config=config,
            documents=empty_docs,
            collection_name=collection_name,
            device=device,
        )

        # Store in cache
        indexers[(collection_name, device)] = indexer

        logging.info(
            f"Successfully created new indexer for collection: {collection_name}"
        )
        return indexer
    except Exception as e:
        raise Exception(f"Unable to create a new indexer: {str(e)}")


def get_indexer(
    collection_name: str, uri: str, db_name: str, device: Optional[str] = None
) -> "Indexer":
    """Get an existing indexer in cached Dict or load from the collection.

    Cached per (collection, device) so each GPU gets its own embedding replica.
    """

    from .index.indexer import Indexer, get_model_from_index
    from .rag.model.dense.base import DenseModelConfig
    from .rag.model.sparse.base import SparseModelConfig

    key = (collection_name, device)
    if key in indexers:
        return indexers[key]

    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=uri, db_name=db_name, enable_sparse=True)

        collections = cast(List[str], client.list_collections())

        if collection_name not in collections:
            return create_new_indexer(collection_name, uri, db_name, device=device)

        # Get model configs from the collection
        dense_config = cast(
            DenseModelConfig,
            get_model_from_index(client, "dense_embedding", collection_name),
        )
        sparse_config = cast(
            SparseModelConfig,
            get_model_from_index(client, "sparse_embedding", collection_name),
        )

        # Create and store the indexer
        indexer = Indexer(
            dense_model_config=dense_config,
            sparse_model_config=sparse_config,
            client=client,
            device=device,
        )

        indexers[key] = indexer

        return indexer
    except Exception as e:
        raise Exception(
            f"Collection {collection_name} not found or could not be loaded: {str(e)}"
        )


def get_retriever_from_config(config: "RetrieverConfig") -> "Retriever":
    """
    Make a retriever from a configuration object (even if a retriever for the same database is already cached).

    Args:
    config, the configuration object for the retriever

    Returns the corresponding Retriever object.
    """

    from .rag.retriever import Retriever

    return Retriever.from_config(config)


def get_retriever(uri: str, db_name: str) -> "Retriever":
    """
    Get an existing retriever in cached Dict or load from the DB uri.

    Args:
    uri, the uri of the database file
    db_name, the name of the database (ignored if the uri is already associated with the retriever)

    Returns the corresponding Retriever object.
    """

    from .rag.retriever import RetrieverConfig

    if uri not in retrievers:
        config = load_config({"db": {"uri": uri, "name": db_name}}, RetrieverConfig)
        retrievers[uri] = get_retriever_from_config(config)

    return retrievers[uri]


def process_files_default(
    temp_dir: str,
    collection_name: str,
    extensions: List[str] = [
        ".pdf",
        ".docx",
        ".pptx",
        ".md",
        ".txt",
        ".xlsx",
        ".xls",
        ".csv",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".mp3",
        ".wav",
        ".aac",
        ".eml",
        ".htm",
        ".html",
    ],
    device: Optional[str] = None,
    output_path: Optional[str] = None,
) -> List["MultimodalSample"]:
    from .process.crawler import Crawler, CrawlerConfig
    from .process.dispatcher import Dispatcher, DispatcherConfig
    from .process.post_processor.pipeline import PPPipeline, PPPipelineConfig

    # Per-job output_path keeps concurrent jobs from overwriting each
    # other if enforced by the caller
    output_path = output_path or f"./tmp/{collection_name}"

    # crawling
    crawler_config = CrawlerConfig(
        root_dirs=[temp_dir],
        # For more efficient processing give only the extensions needed
        supported_extensions=extensions,
        output_path=output_path,
    )
    crawler = Crawler(config=crawler_config)
    crawl_result = crawler.crawl()

    # dispatching the processing
    dispatcher_config = DispatcherConfig(
        output_path=output_path,
        use_fast_processors=False,
        extract_images=True,
        device=device,
    )

    dispatcher = Dispatcher(result=crawl_result, config=dispatcher_config)
    raw_documents = sum(list(dispatcher()), [])

    # post-processing (chunking)
    default_config = {
        "pp_modules": [{"type": "chunker"}],
        "output": {"output_path": output_path},
    }
    config: PPPipelineConfig = load_config(default_config, PPPipelineConfig)
    pipeline = PPPipeline.from_config(config)
    chunked = pipeline(raw_documents)

    return chunked
