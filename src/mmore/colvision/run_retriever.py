import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import APIRouter, FastAPI
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from mmore.profiler import enable_profiling_from_env, profile_function

from ..utils import load_config
from ..ux import (
    progress,
    quiet_noisy_libs,
    setup_logging,
    step_intro,
    step_summary,
)
from .retriever import ColVisionRetriever, ColVisionRetrieverConfig

RETRIEVER_NAME = "ColVision Retrieve"
RETRIEVER_EMOJI = "🔍"
logger = setup_logging(RETRIEVER_NAME, RETRIEVER_EMOJI)


def read_queries(input_file: Path) -> List[str]:
    with open(input_file, "r") as f:
        return [json.loads(line) for line in f]


def save_results(results: List[List[Document]], queries: List[str], output_file: Path):
    """Save retrieval results to a JSON file."""
    formatted_results = [
        {
            "query": query,
            "context": [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ],
        }
        for query, docs in zip(queries, results)
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_results, f, indent=2, ensure_ascii=False)
    logger.debug(f"Saved results to {output_file}")


@profile_function()
def retrieve(
    config_file: str,
    input_file: str,
    output_file: str,
    model_name_override: Optional[str] = None,
):
    """Retrieve documents for specified queries via ColVision-based similarity search."""
    quiet_noisy_libs()
    # Load the config file
    config = load_config(config_file, ColVisionRetrieverConfig)
    if model_name_override:
        config.model_name = model_name_override
        logger.debug(f"Model overridden via CLI: {model_name_override}")

    logger.debug("Loading retriever...")
    retriever = ColVisionRetriever.from_config(config)

    queries = read_queries(Path(input_file))
    step_intro(
        RETRIEVER_NAME,
        RETRIEVER_EMOJI,
        "Find the PDF pages that best match each query",
        [
            f"{len(queries)} queries",
            f"top_k: {config.top_k}",
            f"model: {config.model_name}",
        ],
    )
    start_time = time.time()

    retrieved_docs_for_all_queries = []
    bar = progress(queries, desc="Retrieving", unit="query")
    for query in bar:
        bar.set_postfix_str(str(query)[:40])
        docs_for_query = retriever.invoke(query, k=config.top_k)
        retrieved_docs_for_all_queries.append(docs_for_query)

    time_taken = time.time() - start_time
    save_results(retrieved_docs_for_all_queries, queries, Path(output_file))
    step_summary(
        RETRIEVER_NAME,
        RETRIEVER_EMOJI,
        time_taken,
        {"queries": len(queries), "top_k": config.top_k},
    )


class RetrieverQuery(BaseModel):
    query: str = Field(..., description="Search query text", max_length=1000)
    top_k: int = Field(
        default=3, ge=1, le=100, description="Number of top results to return"
    )


def make_router(
    config_file: str, model_name_override: Optional[str] = None
) -> APIRouter:
    """Create API router with retriever endpoint."""
    quiet_noisy_libs()
    router = APIRouter()

    # Load the config file
    config = load_config(config_file, ColVisionRetrieverConfig)
    if model_name_override:
        config.model_name = model_name_override
        logger.debug(f"Model overridden via CLI: {model_name_override}")

    logger.debug("Loading retriever...")
    retriever_obj = ColVisionRetriever.from_config(config)

    @router.post("/v1/retrieve", tags=["Retrieval"])
    def retriever(query: RetrieverQuery):
        """Query the ColVision retriever."""
        docs_for_query = retriever_obj.invoke(query.query, k=query.top_k)

        docs_info = []
        for doc in docs_for_query:
            meta = doc.metadata
            docs_info.append(
                {
                    "content": doc.page_content,
                    "similarity": meta.get("similarity", 0.0),
                    "rank": meta.get("rank", 0),
                }
            )

        return {"query": query.query, "results": docs_info}

    return router


@profile_function()
def run_api(
    config_file: str,
    host: str,
    port: int,
    model_name_override: Optional[str] = None,
):
    """Run the ColVision retriever API server."""
    router = make_router(config_file, model_name_override=model_name_override)

    config = load_config(config_file, ColVisionRetrieverConfig)
    step_intro(
        RETRIEVER_NAME,
        RETRIEVER_EMOJI,
        "Serve PDF-page search over an API",
        [
            f"http://{host}:{port}",
            f"model: {model_name_override or config.model_name}",
            "endpoint: POST /v1/retrieve",
        ],
    )

    app = FastAPI(
        title="ColVision Retriever API",
        description="""This API is based on the OpenAPI 3.1 specification. You can find out more about Swagger at [https://swagger.io](https://swagger.io).

    ## Overview

    This API defines the ColVision retriever API of mmore, handling:

    1. **Document Retrieval** - Semantic search using ColVision embeddings stored in Milvus.
    2. **PDF Page Search** - Retrieve relevant PDF pages based on query similarity.""",
        version="1.0.0",
    )
    app.include_router(router)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    enable_profiling_from_env()
    parser = argparse.ArgumentParser(
        description="Retrieve documents from local Milvus database using ColVision embeddings."
    )
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to the retriever configuration file.",
    )
    parser.add_argument(
        "--input-file",
        required=False,
        type=str,
        default=None,
        help="Path to the input file of queries. If not provided, the retriever is run in API mode.",
    )
    parser.add_argument(
        "--output-file",
        required=False,
        type=str,
        default=None,
        help="Path to the output file of selected documents. Must be provided together with --input-file.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host on which the API should be run.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port on which the API should be run.",
    )
    args = parser.parse_args()

    if (args.input_file is None) != (args.output_file is None):
        parser.error(
            "Both --input-file and --output-file must be provided together or not at all."
        )

    if args.input_file:  # an input file is provided
        retrieve(args.config_file, args.input_file, args.output_file)
    else:
        run_api(args.config_file, args.host, args.port)
