"""
Vector database retriever using Milvus for efficient similarity search.
Works in conjunction with the Indexer class for document retrieval.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, cast, get_args

import torch
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..index.indexer import DBConfig, get_model_from_index
from ..utils import load_config
from ..ux import loading_model
from .model.dense.base import DenseModel, DenseModelConfig
from .model.sparse.base import SparseModel, SparseModelConfig

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    db: DBConfig = field(default_factory=DBConfig)
    hybrid_search_weight: float = 0.5
    k: int = 1
    collection_name: str = "my_docs"
    use_web: bool = False
    reranker_model_name: Optional[str] = "BAAI/bge-reranker-base"
    jobs_per_gpu: int = 1
    # None below gives by default a queue size of num_gpu * jobs_per_gpu * 10
    max_queue_size: Optional[int] = None


class Retriever(BaseRetriever):
    """Handles similarity-based document retrieval from Milvus."""

    dense_model: Embeddings
    sparse_model: BaseSparseEmbedding
    client: MilvusClient
    hybrid_search_weight: float
    k: int
    use_web: bool
    reranker_model: Optional[PreTrainedModel]
    reranker_tokenizer: Optional[PreTrainedTokenizerBase]

    _search_types = Literal["dense", "sparse", "hybrid"]

    _search_weights = {"dense": 0, "sparse": 1}

    _retrieve_seconds: float = 0.0
    _rerank_seconds: float = 0.0

    # To retrieve the current step within the Retriever (retrieve, rerank or generation)
    _stage_callback: Optional[Callable[[str], None]] = None

    def set_stage_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        self._stage_callback = callback

    def _emit_stage(self, stage: str) -> None:
        if self._stage_callback is not None:
            self._stage_callback(stage)

    def pop_timings(self) -> Tuple[float, float]:
        """Return (retrieve_seconds, rerank_seconds) accumulated so far and reset."""
        retrieve, rerank = self._retrieve_seconds, self._rerank_seconds
        self._retrieve_seconds = 0.0
        self._rerank_seconds = 0.0
        return retrieve, rerank

    @classmethod
    def from_config(cls, config: str | RetrieverConfig):
        if isinstance(config, str):
            config = load_config(config, RetrieverConfig)

        # Init the client
        client = MilvusClient(uri=config.db.uri, db_name=config.db.name)

        if not client.has_collection(config.collection_name):
            raise ValueError(
                f"The Milvus database has not been initialized yet / does not have a collection {config.collection_name}. "
                "Ensure the path is valid with a database that was already populated with the indexer."
            )

        # Init models
        dense_model_config = cast(
            DenseModelConfig,
            get_model_from_index(client, "dense_embedding", config.collection_name),
        )
        dense_model = DenseModel.from_config(dense_model_config)
        logger.debug(f"Loaded dense model: {dense_model_config}")

        sparse_model_config = cast(
            SparseModelConfig,
            get_model_from_index(client, "sparse_embedding", config.collection_name),
        )
        sparse_model = SparseModel.from_config(sparse_model_config)
        logger.debug(f"Loaded sparse model: {sparse_model_config}")

        # Load reranker from Hugging Face
        if config.reranker_model_name:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            with loading_model(f"the reranker model ({config.reranker_model_name})"):
                reranker_tokenizer = AutoTokenizer.from_pretrained(
                    config.reranker_model_name
                )
                reranker_model = AutoModelForSequenceClassification.from_pretrained(
                    config.reranker_model_name
                ).to(device)

            logger.debug(f"Loaded reranker model: {config.reranker_model_name}")
        else:
            reranker_model = reranker_tokenizer = None

        return cls(
            dense_model=dense_model,
            sparse_model=sparse_model,
            client=client,
            hybrid_search_weight=config.hybrid_search_weight,
            k=config.k,
            use_web=config.use_web,
            reranker_model=reranker_model,
            reranker_tokenizer=reranker_tokenizer,
        )

    def compute_query_embeddings(
        self, query: str
    ) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """Compute both dense and sparse embeddings for the query."""
        dense_embedding = [self.dense_model.embed_query(query)]
        sparse_embedding = [self.sparse_model.embed_query(query)]

        return dense_embedding, sparse_embedding

    def retrieve(
        self,
        query: str,
        collection_name: str = "my_docs",
        partition_names: Optional[List[str]] = None,
        min_score: float = -1.0,  # -1.0 is the minimum possible score anyway
        k: int = 1,
        output_fields: List[str] = [
            "text",
            "paragraph_positions",
            "file_path",
        ],
        search_type: str = "hybrid",  # Options: "dense", "sparse", "hybrid"
        document_ids: List[str] = [],  # Optional: candidate doc IDs to restrict search
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar documents for a given query.

        This method computes dense and sparse query embeddings and builds two separate search requests (one for dense and one for sparse). If candidate document IDs are provided, a filter expression is attacked to both search requests to restrict the search to those documents only.

        Args:
            query: Search query string
            collection_name: the Milvus collection to search in
            partition_names: Specific partitions within the collection
            min_score: Minimal similarity score
            k: Number of documents to retrieve
            output_fields: Fields to return in results
            search_type: Type of search to perform ("dense", "sparse", or "hybrid")
            document_ids: Candidate document Ids to filter the search

        Returns:
            The raw search results (a nested list of dictionaries) returned by Milvus
        """
        if k == 0:
            return []

        assert search_type in get_args(self._search_types), (
            f"Invalid search_type: {search_type}. Must be 'dense', 'sparse', or 'hybrid'"
        )
        search_weight = self._search_weights.get(search_type, self.hybrid_search_weight)

        # Combute both dense and sparse embeddings for the query
        dense_embedding, sparse_embedding = self.compute_query_embeddings(query)

        # Build a filter expression if candidate document IDs are provided.
        # The expression will restrict the search to documents with Ids in the given list
        if document_ids:
            # Create a comme-seperated string of quoted document IDs
            ids_str = ",".join(f'"{d}"' for d in document_ids)
            expr = f"document_id in [{ids_str}]"
        else:
            # No filtering if doc_ids is not provided
            expr = None

        # Prepare the search request for the dense embeddings
        # This request searches within the "dense_embedding" field using cosine similarity
        search_param_1 = {
            "data": dense_embedding,  # Query vector
            "anns_field": "dense_embedding",  # Field to search in
            "param": {
                "metric_type": "COSINE",  # This parameter value must be identical to the one used in the collection schema
                "params": {"nprobe": 10},
            },
            "limit": k,
            "expr": expr,
        }

        # Attach the filtering expression if available
        if expr is not None:
            search_param_1["expr"] = expr

        # Prepare the search request for the sparse embeddings.
        # This request searches within the "sparse_embedding" field using the inner product (IP)
        search_param_2 = {
            "data": sparse_embedding,  # Query vector
            "anns_field": "sparse_embedding",  # Field to search in
            "param": {
                "metric_type": "IP",  # This parameter value must be identical to the one used in the collection schema
                "params": {"nprobe": 10},
            },
            "limit": k,
            "expr": expr,
        }

        # Attach the filtering expression if available
        if expr is not None:
            search_param_2["expr"] = expr

        # Create AnnSearchRequest objects from the parameter dictionaries
        request_1 = AnnSearchRequest(**search_param_1)
        request_2 = AnnSearchRequest(**search_param_2)

        # Call the Milvus hybrid_search to perform both searches and then rerank results.
        # The WeightedRanker combined the scores from the dense and sparse searches
        res = self.client.hybrid_search(
            reqs=[request_1, request_2],  # List of AnnSearchRequests
            ranker=WeightedRanker(
                search_weight, 1 - search_weight
            ),  # Reranking strategy
            limit=k,
            output_fields=output_fields,
            collection_name=collection_name,
            partition_names=partition_names,
        )

        # Apply the threshold of min_score
        return list(
            filter(lambda x: x["distance"] >= min_score, res[0])
        )  # 0 because there is only one query

    def batch_retrieve(
        self,
        queries: List[str],
        collection_name: str = "my_docs",
        partition_names: List[str] = [],
        min_score: float = -1.0,  # -1.0 is the minimum possible score anyway
        k: int = 1,
        output_fields: List[str] = [
            "text",
            "paragraph_positions",
            "file_path",
        ],
        search_type: str = "hybrid",
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch retrieve documents for multiple queries.

        Args:
            queries: List of search query strings
            collection_name: Name of the collection in which the research has to be done
            partition_names: Names of the partitions in which the research has to be done
            min_score: Minimal score of the documents to retrieve
            k: Number of documents to retrieve per query
            output_fields: Fields to return in results
            search_type: Type of search to perform

        Returns:
            List of results for each query
        """
        all_results = []
        for query in queries:
            results = self.retrieve(
                query=query,
                collection_name=collection_name,
                partition_names=partition_names,
                min_score=min_score,
                k=k,
                output_fields=output_fields,
                search_type=search_type,
            )
            all_results.append(results)
        return all_results

    def rerank(
        self, query: str, docs: List[Document], batch_size: int = 32
    ) -> List[Document]:
        """Re-rank documents using the reranker model in efficient batches."""
        assert self.reranker_tokenizer is not None
        assert self.reranker_model is not None

        if not docs:
            return []

        scores = []

        # Process documents in batches
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]

            # Prepare query-doc pairs for the batch
            inputs = self.reranker_tokenizer(
                [query] * len(batch_docs),
                [doc.page_content for doc in batch_docs],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.reranker_model.device)

            # Forward pass on the batch
            with torch.no_grad():
                logits = self.reranker_model(**inputs).logits.squeeze(
                    -1
                )  # shape: (batch,)

            # Collect scores for this batch
            scores.extend((doc, score.item()) for doc, score in zip(batch_docs, logits))

        # Sort by reranker score and persist scores for the judge / metrics
        sorted_pairs = sorted(scores, key=lambda x: x[1], reverse=True)
        reranked: List[Document] = []
        for rank_i, (doc, score) in enumerate(sorted_pairs, start=1):
            doc.metadata["rerank_score"] = score
            doc.metadata["rank"] = rank_i
            reranked.append(doc)
        return reranked

    def _get_relevant_documents(
        self,
        query: str | Dict[str, Any],
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieve relevant documents from Milvus. This is necessary for compatibility with LangChain."""

        if isinstance(query, str):
            query_input: str = query
            collection_name: str = kwargs.get("collection_name", "my_docs")
            document_ids: List[str] = kwargs.get("document_ids", [])
        else:
            if "input" not in query:
                raise ValueError("Missing query input")

            query_input: str = query["input"]
            collection_name: str = query.get("collection_name", "my_docs")
            document_ids: List[str] = query.get("document_ids", [])

        partition_names: Optional[List[str]] = kwargs.get("partition_names", None)
        min_score: float = kwargs.get("min_score", -1.0)
        k: int = kwargs.get("k", self.k)

        if k == 0:
            return []

        self._emit_stage("retrieve")
        time_start = time.perf_counter()
        results = self.retrieve(
            query=query_input,
            collection_name=collection_name,
            partition_names=partition_names,
            min_score=min_score,
            k=k,
            document_ids=document_ids,
        )
        retrieve_elapsed = time.perf_counter() - time_start

        def parse_result(result: Dict[str, Any], i: int, offset: int = 0) -> Document:
            return Document(
                page_content=result["entity"]["text"],
                metadata={
                    "id": result["id"],
                    "rank": offset + i + 1,
                    "similarity": result["distance"],
                    "paragraph_positions": result["entity"].get(
                        "paragraph_positions", []
                    ),
                    "file_path": result["entity"].get("file_path", ""),
                },
            )

        def parse_results(
            results: List[Dict[str, Any]], offset: int = 0
        ) -> List[Document]:
            return [parse_result(result, i, offset) for i, result in enumerate(results)]

        if self.use_web:
            web_docs = self._get_web_documents(query_input, max_results=self.k)
            milvus_docs = parse_results(results, len(web_docs))
            docs = web_docs + milvus_docs
        else:
            docs = parse_results(results)

        # Apply reranker
        rerank_elapsed = 0.0
        if self.reranker_model:
            self._emit_stage("rerank")
            time_start = time.perf_counter()
            docs = self.rerank(query_input, docs)
            rerank_elapsed = time.perf_counter() - time_start

        self._retrieve_seconds += retrieve_elapsed
        self._rerank_seconds += rerank_elapsed

        return docs

    def _get_web_documents(self, query: str, max_results: int = 5) -> List[Document]:
        """Fetch additional context from the web via DuckDuckGo."""
        logger.debug("Performing web search...")
        try:
            wrapper = DuckDuckGoSearchAPIWrapper()
            results = wrapper.results(query, max_results=max_results)
            return [
                Document(
                    page_content=result["snippet"],
                    metadata={
                        "source": "duckduckgo",
                        "url": result["link"],
                        "title": result["title"],
                        "rank": i + 1,
                    },
                )
                for i, result in enumerate(results)
            ]
        except Exception as e:
            logger.warning(f"Langchain-DuckDuckGo search failed: {e}")
            return []

    def get_documents_by_ids(
        self, doc_ids: list[str], collection_name: str = "my_docs"
    ) -> list[Document]:
        """
        Fetch documents with the specified IDs from Milvus (if they exist).
        """

        # If no document IDs are provided, return empty list
        if not doc_ids:
            return []

        # Build a comma-seperated string of document IDs, each wrapped in double quotes
        # example: if doc_ids = ["doc1", "doc2"],
        # ids_str becomes '"doc1", "doc2"'
        # This format is required for querying the "id" PK field in Milvus which is a VARCHAR
        ids_str = ",".join(f'"{d}"' for d in doc_ids)
        expr = f"id in [{ids_str}]"

        logger.info(f"Querying Milvus by expr: {expr}")

        results = self.client.query(collection_name, expr, ["id", "text"])

        # If the query returned no results, log a warning
        if not results:
            logger.warning(f"Warning: No documents found for the given IDs: {doc_ids}")
        else:
            # Extract IDs of retrieved documents.
            found_ids = {row["id"] for row in results}
            # Determine which requested IDs were not found (if any)
            missing_ids = [doc_id for doc_id in doc_ids if doc_id not in found_ids]
            if missing_ids:
                logger.warning(
                    f"Warning: The following IDs were not found: {missing_ids}"
                )

        # Convert the returned rows into Document objects
        # Each document contains the text and metadata
        return [
            Document(page_content=row["text"], metadata={"id": row["id"]})
            for row in results
        ]

    def list_files(
        self, collection_name: str, limit: int = 16000
    ) -> List[Dict[str, Any]]:
        """
        List up to ``limit`` unique files currently stored in the database.

        Note:
            By default, ``limit`` is 16000. If there are more files than this in the
            collection, the result will be truncated to at most ``limit`` entries.
            Callers can provide a larger ``limit`` value (or implement pagination at a
            higher level) if they need to enumerate more files.
        Args:
            collection_name: Name of the Milvus collection to query.
            limit: Maximum number of records to retrieve from the collection.
        """

        try:
            results = self.client.query(
                collection_name=collection_name,
                filter='document_id != ""',
                output_fields=["document_id", "filename"],
                limit=limit,
            )

            # Primary change, as requested
            id_to_filename = {}
            for res in results:
                doc_id = res.get("document_id") or res.get("entity", {}).get(
                    "document_id"
                )
                fname = res.get("filename") or res.get("entity", {}).get(
                    "filename", "Unknown"
                )

                if doc_id:
                    id_to_filename[doc_id] = fname

            # list of dictionaries for the API
            return [
                {"id": doc_id, "filename": fname}
                for doc_id, fname in id_to_filename.items()
            ]

        except Exception as e:
            logger.error(f"Error listing files: {e}")
            raise e
