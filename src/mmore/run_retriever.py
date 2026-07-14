import argparse

from dotenv import load_dotenv

load_dotenv()

import json
import re
import time
from pathlib import Path
from typing import List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Query
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from tqdm import tqdm

from mmore.profiler import enable_profiling_from_env, profile_function
from mmore.rag.retriever import Retriever, RetrieverConfig
from mmore.utils import load_config
from mmore.ux import quiet_noisy_libs, setup_logging, step_intro

RETRIEVER_NAME = "Retrieve"
RETRIEVER_EMOJI = "🔍"
logger = setup_logging(RETRIEVER_NAME, RETRIEVER_EMOJI)

load_dotenv()


def read_queries(input_file: Path) -> List[str]:
    with open(input_file, "r") as f:
        return [json.loads(line) for line in f]


def save_results(results: List[List[Document]], queries: List[str], output_file: Path):
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

    # Write to the specified output file
    with open(output_file, "w") as f:
        json.dump(formatted_results, f, indent=2)


@profile_function()
def retrieve(
    config_file: str, input_file: str, output_file: str, document_ids: list[str] = []
):
    """Retrieve documents for specified queries via a vector based similarity search.

    If candidate document IDs are provided, the search is restricted to those documents attaching a filter expression to both dense and sparse search requests. Otherwise, a full collection search is performed
    """

    # Load the config file
    config = load_config(config_file, RetrieverConfig)

    logger.info("Running retriever...")
    retriever = Retriever.from_config(config)
    logger.info("Retriever loaded!")

    # Read queries from the JSONL file
    queries = read_queries(Path(input_file))  # Added missing argument

    # Process document_ids into a list
    doc_ids_list = (
        [doc_id.strip() for doc_id in document_ids if doc_id.strip()]
        if document_ids
        else None
    )

    # Measure time for the retrieval process
    logger.info("Starting document retrieval...")
    start_time = time.time()  # Start timer

    retrieved_docs_for_all_queries = []

    # Call invoke with doc_ids so that the callback manager is used
    for query in tqdm(queries, desc="Retrieving documents", unit="query"):
        docs_for_query = retriever.invoke(query, document_ids=doc_ids_list)
        retrieved_docs_for_all_queries.append(docs_for_query)

    end_time = time.time()  # End timer

    time_taken = end_time - start_time
    logger.info(f"Document retrieval completed in {time_taken:.2f} seconds.")
    logger.info("Retrieved documents!")

    # Save results to output file
    save_results(retrieved_docs_for_all_queries, queries, Path(output_file))
    logger.info(f"Done! Results saved to {output_file}")


class Msg(BaseModel):
    role: str
    content: str


class RetrieverQuery(BaseModel):
    query: str = Field(..., description="Search query")
    fileIds: list[str] = Field(
        default_factory=list,
        description="File IDs to search within (empty = whole collection)",
    )
    maxMatches: int = Field(
        ..., ge=1, description="Maximum number of matches to return"
    )
    minSimilarity: Optional[float] = Field(
        -1.0,
        ge=-1.0,
        le=1.0,
        description="Minimum similarity score for results (-1.0 to 1.0)",
    )


_ID_PATTERN = re.compile(r'^[^"+]+$')


def _chunk_metadata(paragraph_positions) -> Optional[dict]:
    if not paragraph_positions:
        return None
    (fp, fi), (lp, li) = paragraph_positions[0], paragraph_positions[-1]
    return {
        "first": {"page": fp, "paragraph": fi},
        "last": {"page": lp, "paragraph": li},
    }


def make_router(config_file: str) -> APIRouter:
    quiet_noisy_libs()
    router = APIRouter()

    # Load the config file
    config = load_config(config_file, RetrieverConfig)

    logger.debug("Running retriever...")
    retriever_obj = Retriever.from_config(config)
    logger.debug("Retriever loaded!")

    @router.get(
        "/list_files",
        tags=["Files"],
        summary="List files in a collection",
        responses={
            200: {
                "description": "Files currently stored in the collection",
                "content": {
                    "application/json": {
                        "example": [
                            {"id": "doc1", "filename": "report.pdf"},
                            {"id": "doc2", "filename": "notes.md"},
                        ]
                    }
                },
            },
        },
    )
    def list_files(
        collection_name: str, limit: int = Query(default=16000, ge=1, le=100000)
    ):
        """List all files currently in the database."""
        return retriever_obj.list_files(collection_name=collection_name, limit=limit)

    @router.post(
        "/v1/retrieve",
        tags=["Retrieval"],
        summary="Retrieve the most similar chunks for a query",
        responses={
            200: {
                "description": "Matching chunks ordered by similarity",
                "content": {
                    "application/json": {
                        "example": [
                            {
                                "fileId": "doc1",
                                "chunkId": "3",
                                "content": "the matched passage...",
                                "similarity": 0.87,
                                "metadata": {
                                    "first": {"page": 0, "paragraph": 2},
                                    "last": {"page": 0, "paragraph": 2},
                                },
                            }
                        ]
                    }
                },
            },
        },
    )
    def retriever(query: RetrieverQuery):
        """Query the retriever"""

        docs_for_query = retriever_obj.invoke(
            query.query,
            document_ids=query.fileIds,
            k=query.maxMatches,
            min_score=query.minSimilarity,
            collection_name=config.collection_name,
        )

        docs_info = []
        for doc in docs_for_query:
            meta = doc.metadata
            if "+" in meta["id"]:
                fileId, chunkId = meta["id"].rsplit("+", 1)
            else:
                fileId = meta["id"]
                chunkId = None

            docs_info.append(
                {
                    "fileId": fileId,
                    "chunkId": chunkId,
                    "filePath": meta.get("file_path", ""),
                    "content": doc.page_content,
                    "similarity": meta["similarity"],
                    "metadata": _chunk_metadata(meta.get("paragraph_positions")),
                }
            )

        return docs_info

    @router.get(
        "/v1/chunks/{fileId}/{chunkId}",
        tags=["Retrieval"],
        summary="Fetch a chunk's content and metadata by reference",
        responses={
            200: {
                "description": "Chunk content and positional metadata",
                "content": {
                    "application/json": {
                        "example": {
                            "fileId": "doc1",
                            "chunkId": "3",
                            "filename": "report.pdf",
                            "content": "the chunk text...",
                            "metadata": {
                                "first": {"page": 0, "paragraph": 2},
                                "last": {"page": 1, "paragraph": 0},
                            },
                        }
                    }
                },
            },
            400: {"description": "fileId or chunkId contains a forbidden character ('+' or '\"')"},
            404: {"description": "Chunk not found for the given file"},
        },
    )
    def get_chunk(fileId: str, chunkId: str):
        """Fetch a chunk's content and positional metadata by reference."""
        if not _ID_PATTERN.match(fileId) or not _ID_PATTERN.match(chunkId):
            raise HTTPException(400, "fileId and chunkId must not contain '+' or '\"'")
        chunk_ref_literal = json.dumps(f"{fileId}+{chunkId}")
        results = retriever_obj.client.query(
            collection_name=config.collection_name,
            filter=f"id in [{chunk_ref_literal}]",
            output_fields=[
                "text",
                "paragraph_positions",
                "file_path",
                "filename",
            ],
            limit=1,
        )
        if not results:
            raise HTTPException(404, f"Chunk {chunkId} not found for file {fileId}")
        row = results[0]
        entity = row.get("entity", {})
        return {
            "fileId": fileId,
            "chunkId": chunkId,
            "filePath": row.get("file_path") or entity.get("file_path", ""),
            "filename": row.get("filename") or entity.get("filename"),
            "content": row.get("text") or entity.get("text", ""),
            "metadata": _chunk_metadata(
                row.get("paragraph_positions") or entity.get("paragraph_positions")
            ),
        }

    return router


@profile_function()
def run_api(config_file: str, host: str, port: int):
    quiet_noisy_libs()
    config = load_config(config_file, RetrieverConfig)
    step_intro(
        RETRIEVER_NAME,
        RETRIEVER_EMOJI,
        "Serve document search over an API",
        [
            f"http://{host}:{port}",
            f"collection: {config.collection_name}",
            "endpoint: POST /v1/retrieve",
        ],
    )
    router = make_router(config_file)

    app = FastAPI(
        title="mmore Retriever API",
        description="""This API is based on the OpenAPI 3.1 specification. You can find out more about Swagger at [https://swagger.io](https://swagger.io).

    ## Overview

    This API defines the retriever API of mmore, handling:

    1. **File Operations** - Direct file management within mmore.
    2. **Context Retrieval** - Semantic search based on the subset of documents that the user wants.""",
        version="1.0.0",
    )
    app.include_router(router)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    enable_profiling_from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", required=True, help="Path to the retriever configuration file."
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
        help="Path to the output file of selected documents. Must be provided together with --input_file.",
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
