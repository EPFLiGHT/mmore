# tests/test_indexer.py

import json
from unittest.mock import patch

import pytest
import yaml
from conftest import SAMPLE_DOCS, FakeSparseEmbedding
from pymilvus import MilvusClient

from mmore.index.indexer import Indexer
from mmore.rag.model import DenseModelConfig, SparseModelConfig
from mmore.run_index import index
from mmore.type import DocumentMetadata, MultimodalSample


@pytest.fixture
def sample_jsonl(tmp_path):
    path = tmp_path / "sample_docs.jsonl"
    sample_data = [
        {"id": "1", "text": "Document text 1", "modalities": [], "metadata": {}},
        {
            "id": "2",
            "text": "Document text 2",
            "modalities": [],
            "metadata": {"author": "Alice"},
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for entry in sample_data:
            f.write(json.dumps(entry) + "\n")
    return path


@patch("mmore.index.indexer.SparseModel.from_config")
def test_indexer_insert(mock_sparse, tmp_path):
    """Check that documents are written to the Milvus database."""
    mock_sparse.return_value = FakeSparseEmbedding()

    client = MilvusClient(str(tmp_path / "test.db"), enable_sparse=True)
    indexer = Indexer(
        dense_model_config=DenseModelConfig(model_name="debug"),
        sparse_model_config=SparseModelConfig(
            model_name="naver/splade-cocondenser-selfdistil"
        ),
        client=client,
    )

    inserted = indexer.index_documents(SAMPLE_DOCS, collection_name="test_col")

    assert inserted == len(SAMPLE_DOCS)
    stats = client.get_collection_stats("test_col")
    assert int(stats["row_count"]) == len(SAMPLE_DOCS)


@patch("mmore.index.indexer.SparseModel.from_config")
def test_indexer_insert_preserves_metadata(mock_sparse, tmp_path):
    """Check that metadata fields are stored and retrievable."""
    mock_sparse.return_value = FakeSparseEmbedding()

    client = MilvusClient(str(tmp_path / "test.db"), enable_sparse=True)
    indexer = Indexer(
        dense_model_config=DenseModelConfig(model_name="debug"),
        sparse_model_config=SparseModelConfig(
            model_name="naver/splade-cocondenser-selfdistil"
        ),
        client=client,
    )
    indexer.index_documents(SAMPLE_DOCS, collection_name="test_col")

    results = client.query(
        collection_name="test_col",
        filter='id == "doc-2"',
        output_fields=["id", "text", "author"],
    )
    assert len(results) == 1
    assert results[0]["author"] == "Alice"
    assert results[0]["text"] == "The Eiffel Tower stands 330 metres tall."


@patch("mmore.index.indexer.SparseModel.from_config")
def test_indexer_idempotent_collection(mock_sparse, tmp_path):
    """Check that calling index_documents twice on the same collection appends."""
    mock_sparse.return_value = FakeSparseEmbedding()

    client = MilvusClient(str(tmp_path / "test.db"), enable_sparse=True)
    indexer = Indexer(
        dense_model_config=DenseModelConfig(model_name="debug"),
        sparse_model_config=SparseModelConfig(
            model_name="naver/splade-cocondenser-selfdistil"
        ),
        client=client,
    )

    extra = [
        MultimodalSample(
            id="doc-4",
            document_id="doc-4",
            text="A fourth document.",
            modalities=[],
            metadata=DocumentMetadata(),
        )
    ]

    indexer.index_documents(SAMPLE_DOCS, collection_name="test_col")
    indexer.index_documents(extra, collection_name="test_col")

    stats = client.get_collection_stats("test_col")
    assert int(stats["row_count"]) == len(SAMPLE_DOCS) + len(extra)


@patch("mmore.index.indexer.SparseModel.from_config")
def test_run_index(mock_sparse, tmp_path, sample_jsonl):
    """Check that run_index.index() reads config file and write to Milvus DB."""
    mock_sparse.return_value = FakeSparseEmbedding()

    db_path = str(tmp_path / "test.db")
    config = {
        "indexer": {
            "dense_model": {"model_name": "debug", "is_multimodal": False},
            "sparse_model": {
                "model_name": "naver/splade-cocondenser-selfdistil",
                "is_multimodal": False,
            },
            "db": {"uri": db_path, "name": "my_db"},
        },
        "collection_name": "test_col",
        "documents_path": str(sample_jsonl),
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    index(config_file=str(config_path))

    client = MilvusClient(db_path)
    stats = client.get_collection_stats("test_col")
    assert int(stats["row_count"]) == 2  # sample_jsonl has 2 documents


@patch("mmore.index.indexer.SparseModel.from_config")
def test_indexer_error_on_missing_collection(mock_sparse, tmp_path):
    """Check that inserting to a non-existent collection raises an exception."""
    mock_sparse.return_value = FakeSparseEmbedding()

    client = MilvusClient(str(tmp_path / "test.db"), enable_sparse=True)
    indexer = Indexer(
        dense_model_config=DenseModelConfig(model_name="debug"),
        sparse_model_config=SparseModelConfig(
            model_name="naver/splade-cocondenser-selfdistil"
        ),
        client=client,
    )

    doc = MultimodalSample(
        id="x", document_id="x", text="test", modalities=[], metadata=DocumentMetadata()
    )
    with pytest.raises(Exception):
        # Bypasses index_documents (which creates the collection) and inserts directly
        indexer._index_documents([doc], collection_name="nonexistent_collection")
