import argparse
import time
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from mmore.profiler import enable_profiling_from_env, profile_function

from ..utils import load_config
from ..ux import quiet_noisy_libs, setup_logging, step_intro, step_summary
from .milvuscolvision import MilvusColvisionManager

INDEX_NAME = "ColVision Index"
INDEX_EMOJI = "📇"
logger = setup_logging(INDEX_NAME, INDEX_EMOJI)


@dataclass
class MilvusConfig:
    db_path: str = "./milvus_data"
    collection_name: str = "pdf_pages"
    create_collection: bool = False
    metric_type: str = "IP"


@dataclass
class IndexConfig:
    milvus: MilvusConfig
    parquet_path: str


def _get_embedding_dim(df: pd.DataFrame) -> int:
    arr = np.asarray(df.iloc[0]["embedding"])
    if arr.dtype == object:
        arr = np.stack([np.asarray(x) for x in arr])
    return int(arr.shape[-1])


@profile_function()
def index(config_file: Union[IndexConfig, str]):
    """
    Main indexing function.
    Loads embeddings from parquet and inserts them into Milvus local DB.
    """
    quiet_noisy_libs()
    config = load_config(config_file, IndexConfig)

    parquet_path = config.parquet_path
    df = pd.read_parquet(parquet_path)
    logger.debug(f"Indexing {len(df)} rows from {parquet_path}")

    if df.empty:
        logger.warning(f"Parquet {parquet_path} is empty — nothing to index, exiting.")
        return

    step_intro(
        INDEX_NAME,
        INDEX_EMOJI,
        "Store PDF-page data so the pages can be searched",
        [f"{len(df)} rows", f"collection: {config.milvus.collection_name}"],
    )

    dim = _get_embedding_dim(df)
    logger.debug(f"Detected embedding dim={dim} from parquet")

    start = time.time()
    manager = MilvusColvisionManager(
        db_path=config.milvus.db_path,
        collection_name=config.milvus.collection_name,
        dim=dim,
        metric_type=config.milvus.metric_type,
        create_collection=config.milvus.create_collection,
    )
    manager.insert_from_dataframe(df)
    manager.create_index()
    step_summary(
        INDEX_NAME,
        INDEX_EMOJI,
        time.time() - start,
        {"indexed": f"{len(df)} rows", "collection": config.milvus.collection_name},
    )


if __name__ == "__main__":
    enable_profiling_from_env()
    parser = argparse.ArgumentParser(
        description="Index ColVision PDF embeddings into a local Milvus database."
    )
    parser.add_argument(
        "--config-file", required=True, help="Path to the YAML config file."
    )
    args = parser.parse_args()

    index(args.config_file)
