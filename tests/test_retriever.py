"""
Integration tests for the Retriever: indexes documents into Milvus Lite,
then queries via the actual hybrid_search pipeline.
"""

from unittest.mock import MagicMock, patch

import pytest
from conftest import CHUNKED_CORPUS, CHUNKED_SAMPLE_DOCS, FakeSparseEmbedding
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
from pymilvus import MilvusClient

from mmore.index.indexer import Indexer
from mmore.rag.model import DenseModelConfig, SparseModelConfig
from mmore.rag.retriever import Retriever

_COLLECTION = "test_col"


@pytest.fixture(scope="module")
def populated_db(tmp_path_factory):
    """Initial Milvus database population."""
    db_path = str(tmp_path_factory.mktemp("retriever_db") / "test.db")

    with patch(
        "mmore.index.indexer.SparseModel.from_config",
        return_value=FakeSparseEmbedding(),
    ):
        client = MilvusClient(db_path, enable_sparse=True)
        indexer = Indexer(
            dense_model_config=DenseModelConfig(model_name="debug"),
            sparse_model_config=SparseModelConfig(
                model_name="naver/splade-cocondenser-selfdistil"
            ),
            client=client,
        )
        indexer.index_documents(CHUNKED_SAMPLE_DOCS, collection_name=_COLLECTION)

    return db_path


@pytest.fixture(scope="module")
def retriever(populated_db):
    """Retriever pointing at the populated DB."""
    client = MilvusClient(populated_db, enable_sparse=True)
    return Retriever(
        dense_model=FakeEmbeddings(size=2048),
        sparse_model=FakeSparseEmbedding(),
        client=client,
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=None,
        reranker_tokenizer=None,
    )


@pytest.fixture
def small_batch_retriever(retriever, monkeypatch):
    """Same retriever, with the Milvus row iterator set below the number of chunks
    so that listing the files has to span several batches."""
    monkeypatch.setattr("mmore.rag.retriever._MILVUS_QUERY_MAX_ROWS", 2)
    return retriever


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------


def test_retrieve_hybrid_returns_results(retriever):
    results = retriever.retrieve("capital of France", collection_name=_COLLECTION, k=2)
    assert 1 <= len(results) <= 2
    assert all("text" in r["entity"] for r in results)
    assert all("distance" in r for r in results)


def test_retrieve_dense_only(retriever):
    results = retriever.retrieve(
        "Eiffel Tower", collection_name=_COLLECTION, k=2, search_type="dense"
    )
    assert 1 <= len(results) <= 2


def test_retrieve_sparse_only(retriever):
    results = retriever.retrieve(
        "vector database", collection_name=_COLLECTION, k=2, search_type="sparse"
    )
    assert 1 <= len(results) <= 2


def _query_vectors(retriever, query):
    """A dense query vector equal to a stored chunk's own vector, so the
    dense-only top 1 is known, and that chunk is not the sparse-only top 1."""
    sparse = [retriever.sparse_model.embed_query(query)]
    sparse_top = retriever.client.search(
        collection_name=_COLLECTION,
        data=sparse,
        anns_field="sparse_embedding",
        search_params={"metric_type": "IP"},
        limit=1,
    )[0][0]["id"]
    rows = retriever.client.query(
        collection_name=_COLLECTION,
        filter=f'id != "{sparse_top}"',
        output_fields=["dense_embedding"],
        limit=1,
    )
    return rows[0]["id"], sparse_top, [rows[0]["dense_embedding"]], sparse


@pytest.mark.parametrize("search_type", ["dense", "sparse"])
def test_single_search_type_ranks_by_its_own_model(retriever, search_type):
    dense_top, sparse_top, dense, sparse = _query_vectors(
        retriever, "capital of France"
    )
    with patch.object(
        Retriever, "compute_query_embeddings", return_value=(dense, sparse)
    ):
        results = retriever.retrieve(
            "capital of France",
            collection_name=_COLLECTION,
            k=1,
            search_type=search_type,
        )
    assert results[0]["id"] == (dense_top if search_type == "dense" else sparse_top)


def test_retrieve_k_zero_returns_empty(retriever):
    results = retriever.retrieve("anything", collection_name=_COLLECTION, k=0)
    assert results == []


def test_retrieve_min_score_filters_all(retriever):
    # FakeEmbeddings cosine scores are well below 2.0
    results = retriever.retrieve(
        "capital", collection_name=_COLLECTION, k=3, min_score=2.0
    )
    assert results == []


def test_retrieve_result_ids_are_known_docs(retriever):
    known_ids = {d.id for d in CHUNKED_SAMPLE_DOCS}
    results = retriever.retrieve("Paris", collection_name=_COLLECTION, k=3)
    for r in results:
        assert r["id"] in known_ids


# ---------------------------------------------------------------------------
# batch_retrieve()
# ---------------------------------------------------------------------------


def test_batch_retrieve_returns_one_list_per_query(retriever):
    queries = ["Paris", "Eiffel Tower", "open-source"]
    results = retriever.batch_retrieve(queries, collection_name=_COLLECTION, k=2)
    assert len(results) == len(queries)
    assert all(isinstance(r, list) for r in results)


def test_batch_retrieve_empty_queries(retriever):
    results = retriever.batch_retrieve([], collection_name=_COLLECTION, k=2)
    assert results == []


# ---------------------------------------------------------------------------
# get_documents_by_ids()
# ---------------------------------------------------------------------------


def test_get_documents_by_ids_returns_correct_docs(retriever):
    docs = retriever.get_documents_by_ids(
        ["doc-paris+1", "doc-milvus+0"], collection_name=_COLLECTION
    )
    assert len(docs) == 2
    returned_ids = {d.metadata["id"] for d in docs}
    assert returned_ids == {"doc-paris+1", "doc-milvus+0"}
    assert all(isinstance(d, Document) for d in docs)


def test_get_documents_by_ids_empty_input(retriever):
    docs = retriever.get_documents_by_ids([], collection_name=_COLLECTION)
    assert docs == []


def test_get_documents_by_ids_unknown_id(retriever):
    docs = retriever.get_documents_by_ids(
        ["nonexistent-id"], collection_name=_COLLECTION
    )
    assert docs == []


# ---------------------------------------------------------------------------
# list_files()
# ---------------------------------------------------------------------------


def test_list_files_returns_all_documents(retriever):
    files = retriever.list_files(collection_name=_COLLECTION)
    returned_ids = {f["id"] for f in files}
    expected_ids = {d.document_id for d in CHUNKED_SAMPLE_DOCS}
    assert returned_ids == expected_ids


@pytest.mark.parametrize("limit", range(1, len(CHUNKED_CORPUS) + 1))
def test_list_files_limit_counts_files(small_batch_retriever, limit):
    """The limit counts distinct files, not chunks."""
    files = small_batch_retriever.list_files(collection_name=_COLLECTION, limit=limit)
    ids = [f["id"] for f in files]

    assert len(ids) == limit
    assert len(set(ids)) == limit
    assert set(ids) <= set(CHUNKED_CORPUS)
    assert ids == sorted(ids)


def test_list_files_is_sorted_by_id(small_batch_retriever):
    files = small_batch_retriever.list_files(collection_name=_COLLECTION)

    assert [f["id"] for f in files] == sorted(f["id"] for f in files)


def test_list_files_returns_all_files_across_batches(small_batch_retriever):
    files = small_batch_retriever.list_files(collection_name=_COLLECTION)

    assert {f["id"]: f["filename"] for f in files} == {
        document_id: f"{document_id}.txt" for document_id in CHUNKED_CORPUS
    }


# ---------------------------------------------------------------------------
# _get_relevant_documents()
# ---------------------------------------------------------------------------


def test_get_relevant_documents_returns_documents(retriever):
    docs = retriever._get_relevant_documents(
        "France", run_manager=MagicMock(), collection_name=_COLLECTION, k=2
    )
    assert all(isinstance(d, Document) for d in docs)
    assert len(docs) <= 2


def test_get_relevant_documents_dict_input(retriever):
    docs = retriever._get_relevant_documents(
        {"input": "Eiffel Tower", "collection_name": _COLLECTION},
        run_manager=MagicMock(),
        k=2,
    )
    assert all(isinstance(d, Document) for d in docs)


def test_get_relevant_documents_k_zero(retriever):
    docs = retriever._get_relevant_documents(
        "anything", run_manager=MagicMock(), collection_name=_COLLECTION, k=0
    )
    assert docs == []


def test_get_relevant_documents_missing_input_key(retriever):
    with pytest.raises(ValueError, match="Missing query input"):
        retriever._get_relevant_documents(
            {"collection_name": _COLLECTION},
            run_manager=MagicMock(),
        )
