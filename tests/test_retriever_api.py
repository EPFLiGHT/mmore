from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from mmore.rag.retriever import RetrieverConfig


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.client = MagicMock()
    return retriever


@pytest.fixture
def client(mock_retriever):
    from mmore.run_retriever import make_router

    fake_config = RetrieverConfig(collection_name="test_docs")
    with (
        patch("mmore.run_retriever.load_config", return_value=fake_config),
        patch("mmore.run_retriever.Retriever.from_config", return_value=mock_retriever),
    ):
        app = FastAPI()
        app.include_router(make_router("dummy_config.yaml"))
        yield TestClient(app)


def test_get_chunk_flat_row_shape(client, mock_retriever):
    """200 OK when Milvus returns a flat row with text/paragraph_positions at top level."""
    mock_retriever.client.query.return_value = [
        {
            "text": "hello world",
            "paragraph_positions": [[1, 0], [1, 2], [2, 1]],
            "file_path": "papers/example.pdf",
            "filename": "demo.pdf",
        }
    ]

    response = client.get("/v1/chunks/file-123/chunk-7")

    assert response.status_code == 200
    assert response.json() == {
        "fileId": "file-123",
        "chunkId": "chunk-7",
        "filePath": "papers/example.pdf",
        "filename": "demo.pdf",
        "content": "hello world",
        "metadata": {
            "first": {"page": 1, "paragraph": 0},
            "last": {"page": 2, "paragraph": 1},
        },
    }
    call_kwargs = mock_retriever.client.query.call_args.kwargs
    assert call_kwargs["collection_name"] == "test_docs"
    assert call_kwargs["filter"] == 'id in ["file-123+chunk-7"]'
    assert call_kwargs["output_fields"] == [
        "text",
        "paragraph_positions",
        "file_path",
        "filename",
    ]
    assert call_kwargs["limit"] == 1


def test_get_chunk_no_paragraph_positions_returns_null_metadata(client, mock_retriever):
    """metadata is null when paragraph_positions is missing or empty."""
    mock_retriever.client.query.return_value = [
        {"text": "no positions", "paragraph_positions": []}
    ]

    response = client.get("/v1/chunks/file-123/chunk-7")

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] is None
    assert body["content"] == "no positions"
    assert body["metadata"] is None
    # missing file_path should fall back to empty string, not crash
    assert body["filePath"] == ""


def test_get_chunk_404_when_not_found(client, mock_retriever):
    """404 when Milvus returns no matching row."""
    mock_retriever.client.query.return_value = []

    response = client.get("/v1/chunks/file-123/missing-chunk")

    assert response.status_code == 404
    assert "missing-chunk" in response.json()["detail"]


@pytest.mark.parametrize(
    "path",
    [
        "/v1/chunks/bad+file/chunk-7",  # '+' in fileId
        "/v1/chunks/file-123/bad+chunk",  # '+' in chunkId
        "/v1/chunks/file%22123/chunk-7",  # '"' in fileId (URL-encoded)
        "/v1/chunks/file-123/bad%22chunk",  # '"' in chunkId
    ],
)
def test_get_chunk_400_rejects_invalid_id_chars(client, mock_retriever, path):
    """Per OpenAPI spec, fileId/chunkId must reject '+' and '"' without querying Milvus."""
    response = client.get(path)

    assert response.status_code == 400
    mock_retriever.client.query.assert_not_called()


def test_retrieve_returns_gateway_contract_shape(client, mock_retriever):
    """POST /v1/retrieve must return the shape consumed by moove-gateway's
    MmoreRetrievalResult: {fileId, chunkId, content, similarity, metadata}."""
    mock_retriever.invoke.return_value = [
        Document(
            page_content="first chunk",
            metadata={
                "id": "file-1+chunk-0",
                "similarity": 0.91,
                "paragraph_positions": [[1, 0], [1, 3]],
                "file_path": "papers/one.pdf",
            },
        ),
        Document(
            page_content="second chunk",
            metadata={
                "id": "file-2+chunk-5",
                "similarity": 0.42,
                "paragraph_positions": [],
                "file_path": "papers/two.pdf",
            },
        ),
    ]

    response = client.post(
        "/v1/retrieve",
        json={
            "query": "what is X?",
            "fileIds": ["file-1", "file-2"],
            "maxMatches": 2,
            "minSimilarity": 0.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body == [
        {
            "fileId": "file-1",
            "chunkId": "chunk-0",
            "filePath": "papers/one.pdf",
            "content": "first chunk",
            "similarity": 0.91,
            "metadata": {
                "first": {"page": 1, "paragraph": 0},
                "last": {"page": 1, "paragraph": 3},
            },
        },
        {
            "fileId": "file-2",
            "chunkId": "chunk-5",
            "filePath": "papers/two.pdf",
            "content": "second chunk",
            "similarity": 0.42,
            "metadata": None,
        },
    ]
    invoke_kwargs = mock_retriever.invoke.call_args.kwargs
    assert invoke_kwargs["document_ids"] == ["file-1", "file-2"]
    assert invoke_kwargs["k"] == 2
    assert invoke_kwargs["min_score"] == 0.0


def test_retrieve_handles_id_without_chunk_suffix(client, mock_retriever):
    """When a document id has no '+' separator, chunkId is returned as null."""
    mock_retriever.invoke.return_value = [
        Document(
            page_content="whole doc",
            metadata={
                "id": "file-no-chunks",
                "similarity": 0.5,
                "paragraph_positions": [],
            },
        )
    ]

    response = client.post(
        "/v1/retrieve",
        json={"query": "q", "fileIds": [], "maxMatches": 1, "minSimilarity": -1.0},
    )

    assert response.status_code == 200
    [item] = response.json()
    assert item["fileId"] == "file-no-chunks"
    assert item["chunkId"] is None
    assert item["metadata"] is None
    # missing file_path should not crash; default to empty string
    assert item["filePath"] == ""


def test_retrieve_handles_id_with_multiple_plus_separators(client, mock_retriever):
    """When a doc id contains multiple '+', split from the right so the last
    segment is the chunk index and everything before is the file id. Avoids a
    500 for ids where the user-supplied document id itself contains '+'."""
    mock_retriever.invoke.return_value = [
        Document(
            page_content="c",
            metadata={
                "id": "file+with+plus+42",
                "similarity": 0.5,
                "paragraph_positions": [],
            },
        )
    ]

    response = client.post(
        "/v1/retrieve",
        json={"query": "q", "fileIds": [], "maxMatches": 1, "minSimilarity": -1.0},
    )

    assert response.status_code == 200
    [item] = response.json()
    assert item["fileId"] == "file+with+plus"
    assert item["chunkId"] == "42"
