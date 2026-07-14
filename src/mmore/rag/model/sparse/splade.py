from typing import Dict, List

import scipy
import torch
from langchain_milvus.utils.sparse import BaseSparseEmbedding


class SpladeSparseEmbedding(BaseSparseEmbedding):
    """Sparse embedding model based on Splade.

    This class uses the Splade embedding model in Milvus model to implement sparse vector embedding.
    This model requires pymilvus[model] to be installed.
    `pip install pymilvus[model]`
    For more information please refer to:
    https://milvus.io/docs/embed-with-splade.md
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-selfdistil",
        device: str | None = None,
    ):
        from pymilvus.model.sparse import SpladeEmbeddingFunction  # type: ignore

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.splade = SpladeEmbeddingFunction(model_name=model_name, device=self.device)

    def embed_query(self, query: str) -> Dict[int, float]:
        res = self.splade.encode_queries([query])
        # res[0] because res has one row per query and there is only one query
        # conversion from coo_array to csr_array is needed with new version of pymilvus
        res_as_dict: Dict[int, float] = self._sparse_row_to_dict(res[0])
        return res_as_dict

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        res = self.splade.encode_documents(texts)
        res_as_dicts: List[Dict[int, float]] = [
            self._sparse_row_to_dict(row) for row in res
        ]
        return res_as_dicts

    def _sparse_row_to_dict(self, row: scipy.sparse.coo_array) -> Dict[int, float]:
        """Convert a sparse row to a dict of index.

        Conversion from coo_array to csr_array is needed with new versions of pymilvus.
        """
        csr_row = row.tocsr()
        return {k: v for k, v in zip(csr_row.indices.tolist(), csr_row.data.tolist())}
