import concurrent.futures
import logging
from pathlib import Path
from typing import Any, Union, cast

import numpy as np
import pandas as pd
from pymilvus import DataType, MilvusClient

from ..ux import is_verbose, progress

logger = logging.getLogger(__name__)


class MilvusColvisionManager:
    """
    Handles all Milvus operations for ColVision embeddings (both indexing and retrieval).
    Uses a local Milvus storage by default.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "./milvus_data",
        collection_name: str = "pdf_pages",
        dim: int = 128,
        metric_type: str = "IP",
        create_collection: bool = False,
    ):
        """
        Initialize a local Milvus database connection and collection handler.
        """
        self.uri = str(Path(db_path).resolve())
        self.collection_name = collection_name
        self.dim = dim
        self.metric_type = metric_type

        logger.debug(f"Connecting to Milvus local instance at {self.uri}")
        self.client = MilvusClient(uri=self.uri)

        if create_collection:
            self.create_collection(overwrite=True)
            logger.debug(f"Created new collection '{self.collection_name}'")
        elif self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)
            logger.debug(f"Loaded existing collection '{self.collection_name}'")
        else:
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist. "
                "Use create_collection=True to create it."
            )

    def create_collection(self, overwrite: bool = False):
        """
        Create a new Milvus collection with ColVision embedding schema.
        """
        if overwrite and self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.debug(f"Dropped existing collection '{self.collection_name}'")

        schema = self.client.create_schema(auto_id=True)
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("pdf_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )
        logger.debug(f"Created collection schema for '{self.collection_name}'")

    def create_index(self):
        """
        Create a vector index on the embedding field.
        Should be called AFTER data insertion for better performance.
        """
        # Check if index already exists
        try:
            index_info = self.client.list_indexes(self.collection_name)
            if index_info:
                logger.debug(f"Index already exists: {index_info}")
                return True
        except Exception as e:
            logger.warning(f"Could not check existing indexes: {e}")

        logger.debug("Creating vector index on 'embedding' field...")

        # Create index parameters
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_name="embedding_index",
            index_type="FLAT",
            metric_type=self.metric_type,
        )

        # Create index
        try:
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params,
            )
            logger.debug("Created vector index on 'embedding' field")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False

    def insert_from_dataframe(self, df: pd.DataFrame, batch_size: int = 500):
        """
        Insert all subvectors from each row of the DataFrame into Milvus in batches.
        """
        required_cols = {"pdf_path", "page_number", "embedding"}
        if not required_cols <= set(df.columns):
            raise ValueError(
                f"DataFrame missing required columns: {required_cols - set(df.columns)}"
            )

        logger.debug(f"Preparing {len(df)} pages...")

        data = []
        for row in progress(
            df.itertuples(index=False),
            total=len(df),
            desc="Preparing vectors",
            unit="page",
        ):
            emb = getattr(row, "embedding")

            if isinstance(emb, np.ndarray) and emb.dtype == object:
                emb = np.stack(cast(Any, emb))
            elif (
                isinstance(emb, list)
                and len(emb) > 0
                and isinstance(emb[0], np.ndarray)
            ):
                emb = np.stack(emb)
            elif isinstance(emb, list):
                emb = np.array(emb, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb[np.newaxis, :]

            for vec in emb:
                data.append(
                    {
                        "pdf_path": getattr(row, "pdf_path"),
                        "page_number": int(getattr(row, "page_number")),
                        "embedding": np.asarray(vec, dtype=np.float32).tolist(),
                    }
                )

        # Validate embedding dimensions
        for i, row in enumerate(data):
            vlen = len(row["embedding"])
            assert vlen == self.dim, (
                f"Row {i}: got embedding len={vlen}, expected {self.dim}"
            )

        total_vecs = len(data)
        logger.debug(f"Inserting {total_vecs} vectors in batches of {batch_size}...")

        for i in progress(
            range(0, total_vecs, batch_size), desc="Inserting", unit="batch"
        ):
            batch = data[i : i + batch_size]
            try:
                self.client.insert(self.collection_name, batch)
            except Exception as e:
                logger.error(f"Failed to insert batch {i // batch_size}: {e}")
                raise

        logger.debug(f"Insert complete: {total_vecs} vectors inserted.")

    def search_embeddings(self, query_embeddings, top_k=3, max_workers=4):
        """
        Search for similar embeddings using vector similarity search.
        """
        # Handle query shape
        arr = np.asarray(query_embeddings, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.squeeze(0)
        elif arr.ndim == 1:
            arr = arr[np.newaxis, :]

        # Perform initial vector search
        results = self.client.search(
            collection_name=self.collection_name,
            data=arr.tolist(),
            anns_field="embedding",
            limit=top_k * 5,
            # As each page is embedded as a multi vector, each page is represented multiple times inside Milvus, each with one column of the multi vector
            # Thus there can be duplicates in the ids, and top_k * 5 allows us to ensure the retrieval of top_k distinct pages
            output_fields=["pdf_path", "page_number"],
            search_params={"metric_type": self.metric_type, "params": {}},
        )

        # Collect unique candidate pages
        candidates = set()
        for hits in results:
            for hit in hits:
                entity_dict = hit.get("entity", {})
                key = (entity_dict.get("pdf_path"), entity_dict.get("page_number"))
                candidates.add(key)

        logger.debug(f"Found {len(candidates)} candidate pages for reranking.")

        def rerank_page(pdf_path, page_number, query_vecs):
            # Get all subvectors for this page
            docs = self.client.query(
                collection_name=self.collection_name,
                filter=f'pdf_path == "{pdf_path}" and page_number == {int(page_number)}',
                output_fields=["embedding", "pdf_path"],
                limit=10000,
            )
            if not docs:
                return (None, pdf_path, page_number)

            doc_vecs = np.vstack([d["embedding"] for d in docs]).astype(np.float32)
            # Cast to a Python float: the score flows into Document.metadata and is
            # later JSON-serialized by run_retriever.save_results(); np.float32 is not
            # JSON-serializable and would crash the whole retrieve subprocess.
            score = float(np.dot(query_vecs, doc_vecs.T).max(1).sum())
            return (score, pdf_path, page_number)

        reranked = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(rerank_page, pdf_path, page_number, arr)
                for (pdf_path, page_number) in candidates
            ]
            for f in progress(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Reranking",
                disable=not is_verbose(),
            ):
                try:
                    score, pdf_path, page_number = f.result()
                    if score is not None:
                        reranked.append(
                            {
                                "pdf_path": pdf_path,
                                "page_number": page_number,
                                "score": score,
                            }
                        )
                except Exception as e:
                    logger.error(f"Rerank failed: {e}")

        # Sort and return top-k pages
        reranked.sort(key=lambda x: x["score"], reverse=True)
        for i, item in enumerate(reranked):
            item["rank"] = i + 1
        return reranked[:top_k]

    def drop_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.debug(f"Dropped collection '{self.collection_name}'")
        else:
            logger.warning(f"Collection '{self.collection_name}' not found.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        if hasattr(self, "client"):
            self.client.close()
            logger.debug("Closed Milvus client connection")
