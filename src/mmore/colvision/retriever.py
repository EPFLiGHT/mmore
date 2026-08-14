"""
ColVision retriever that can be used with the general RAG pipeline.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from colpali_engine.utils.torch_utils import ListDataset
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from torch.utils.data import DataLoader

from .milvuscolvision import MilvusColvisionManager
from .model_utils import get_device, load_model_and_processor

logger = logging.getLogger(__name__)


@dataclass
class ColVisionRetrieverConfig:
    """Configuration for ColVision retriever."""

    db_path: str = "./milvus_data"
    collection_name: str = "pdf_pages"
    model_name: str = "vidore/colqwen2.5-v0.2"
    top_k: int = 3
    max_workers: int = 4
    metric_type: str = "IP"
    text_parquet_path: Optional[str] = None


def embed_queries(texts: List[str], model, processor) -> List[np.ndarray]:
    """
    Generate ColVision embeddings for text queries.

    Args:
        texts: List of query strings to embed
        model: ColVision model instance
        processor: ColVision processor instance

    Returns:
        List of numpy arrays containing query embeddings
    """
    dataloader = DataLoader(
        dataset=ListDataset(texts),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x),
    )

    vectors = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            emb = model(**batch_query)
            vectors.extend(list(emb.to("cpu").unbind()))
    return [v.float().numpy() for v in vectors]


def load_text_mapping(text_parquet_path: Optional[str]) -> Optional[Dict[tuple, str]]:
    """
    Load text mapping from parquet file.
    Returns a dictionary mapping (pdf_path, page_number) to text.
    """
    if text_parquet_path is None:
        return None

    text_path = Path(text_parquet_path)
    if not text_path.exists():
        logger.warning(f"Text parquet file not found: {text_path}")
        return None

    try:
        df = pd.read_parquet(text_path)
        # Create a mapping from (pdf_path, page_number) to text
        text_map = dict(
            zip(zip(df["pdf_path"], df["page_number"].astype(int)), df["text"])
        )
        logger.info(f"Loaded text mapping for {len(text_map)} pages")
        return text_map
    except Exception as e:
        logger.error(f"Failed to load text mapping: {e}")
        raise


class ColVisionRetriever(BaseRetriever):
    """
    ColVision-based retriever that can be used with the RAG pipeline.
    Returns Document objects with text content from PDF pages.
    """

    model: Any
    processor: Any
    manager: MilvusColvisionManager
    config: ColVisionRetrieverConfig
    text_map: Optional[Dict[tuple, str]]

    def __init__(
        self,
        model: Any,
        processor: Any,
        manager: MilvusColvisionManager,
        config: ColVisionRetrieverConfig,
        text_map: Optional[Dict[tuple, str]],
    ):
        """Initialize ColVisionRetriever. Use from_config() to create instances."""
        super().__init__(
            **{
                "model": model,
                "processor": processor,
                "manager": manager,
                "config": config,
                "text_map": text_map,
            }
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        Returns Document objects compatible with LangChain RAG pipeline.

        Args:
            query: Search query string
            run_manager: Optional callback manager
            **kwargs: Additional arguments (e.g., k for top-k results)

        Returns:
            List of Document objects with page_content (text) and metadata
        """
        top_k = kwargs.get("k", self.config.top_k)

        # Embed query and search
        vecs = embed_queries([query], self.model, self.processor)[0]
        results = self.manager.search_embeddings(
            vecs, top_k=top_k, max_workers=self.config.max_workers
        )

        # Add text content to results
        if self.text_map is not None:
            for result in results:
                pdf_path = result.get("pdf_path")
                page_number = result.get("page_number")
                if pdf_path and page_number:
                    key = (pdf_path, int(page_number))
                    result["text"] = self.text_map.get(key, "")
                else:
                    result["text"] = ""
        else:
            for result in results:
                result["text"] = ""

        # Convert to Document objects
        documents = []
        for i, result in enumerate(results):
            pdf_path = result.get("pdf_path", "")
            pdf_name = Path(pdf_path).name if pdf_path else ""
            doc = Document(
                page_content=result.get("text", ""),
                metadata={
                    "pdf_name": pdf_name,
                    "pdf_path": pdf_path,
                    "page_number": result.get("page_number"),
                    "rank": result.get("rank", i + 1),
                    "similarity": result.get("score"),
                },
            )
            documents.append(doc)

        return documents

    @classmethod
    def from_config(cls, config: ColVisionRetrieverConfig):
        """
        Create a ColVisionRetriever from a configuration.

        Args:
            config: ColVision retriever configuration

        Returns:
            ColVisionRetriever instance
        """
        # Load model and processor
        device = get_device()
        model, processor = load_model_and_processor(config.model_name, device)

        # Initialize manager — dim is taken from the loaded model itself
        manager = MilvusColvisionManager(
            db_path=config.db_path,
            collection_name=config.collection_name,
            dim=model.dim,
            metric_type=config.metric_type,
            create_collection=False,
        )

        # Load text mapping
        text_map = load_text_mapping(config.text_parquet_path)

        return cls(
            model=model,
            processor=processor,
            manager=manager,
            config=config,
            text_map=text_map,
        )
