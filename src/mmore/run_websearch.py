# mmore/run_websearch.py

import argparse
import time
from dataclasses import dataclass
from typing import Optional, Union

try:
    import torch
except ImportError:
    torch = None

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from mmore.profiler import enable_profiling_from_env, profile_function
from mmore.rag.pipeline import RAGPipeline
from mmore.run_rag import APIConfig, LocalConfig, RAGInferenceConfig
from mmore.utils import load_config
from mmore.ux import Spinner, quiet_noisy_libs
from mmore.websearchRAG.config import WebsearchConfig
from mmore.websearchRAG.logging_config import logger
from mmore.websearchRAG.pipeline import WebsearchPipeline

load_dotenv()


# CUDA tweaks for best perf
if torch is not None and torch.cuda.is_available():
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


@dataclass
class WebsearchInferenceConfig:
    websearch: WebsearchConfig
    mode_args: Optional[Union[LocalConfig, APIConfig]] = None

    def __post_init__(self):
        if self.websearch.mode == "api" and self.mode_args is None:
            self.mode_args = APIConfig()


@profile_function()
def run_websearch(config_file):
    quiet_noisy_libs()
    cfg = load_config(config_file, WebsearchInferenceConfig)
    ws = cfg.websearch
    if ws.mode == "local":
        with Spinner("Loading websearch pipeline"):
            pipeline = WebsearchPipeline(config=ws)
        logger.debug("Running Websearch pipeline in LOCAL mode...")
        start = time.time()
        pipeline.run()
        logger.debug(f"Completed in {time.time() - start:.2f}s")

    elif ws.mode == "api":
        logger.info("Starting Websearch pipeline in API mode...")
        app = create_api(cfg)
        uvicorn.run(app, host="0.0.0.0", port=8000)

    else:
        raise ValueError(f"Unknown mode: {ws.mode!r}. Must be 'local' or 'api'.")


class QueryInput(BaseModel):
    input: str = Field(..., description="The user query")
    collection_name: str = Field(
        "my_docs", description="The collection to search if use_rag set to True"
    )


class WebQuery(BaseModel):
    query: QueryInput = Field(
        ..., description="Search query with input and optional collection name"
    )
    use_rag: bool = Field(False, description="Include RAG context", examples=[True])
    use_summary: bool = Field(
        True, description="Enable subquery summary", examples=[False]
    )


def create_api(config: WebsearchInferenceConfig):
    app = FastAPI(
        title="mmore Websearch API",
        description="""This API is based on the OpenAPI 3.1 specification. You can find out more about Swagger at [https://swagger.io](https://swagger.io).

## Overview

This API defines the retriever API of mmore, handling:

1. **File Operations** - Direct file management within mmore.
2. **Rag and websearch** - Search based on the query/documents.""",
        version="1.0.0",
    )

    logger.info("Websearch loaded!")

    @app.post("/websearch")
    # query = query parameter
    def websearch(query: WebQuery):
        pipeline = WebsearchPipeline(config=config.websearch)

        if query.use_rag:
            logger.debug("Launch RAG")
            config_rag = load_config(
                config.websearch.rag_config_path, RAGInferenceConfig
            )
            logger.debug("Creating the RAG Pipeline...")
            rag_pp = RAGPipeline.from_config(config_rag.rag)
            data = rag_pp([query.query.dict()], return_dict=True)
            logger.debug("RAG done")
            logger.debug("##RAG## %s", data)
        else:
            data = query.query

        logger.debug("Launch websearch")

        answers = pipeline.run_api(query.use_rag, query.use_summary, data)
        logger.debug("Websearch done")

        return answers

    return app


if __name__ == "__main__":
    enable_profiling_from_env()
    parser = argparse.ArgumentParser(
        description="Run the Websearch (+ optional RAG) pipeline."
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to the Websearch configuration file (YAML).",
    )
    args = parser.parse_args()
    run_websearch(args.config_file)
