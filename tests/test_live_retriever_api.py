"""
Integration tests for the live retrieval API using a Milvus DB:
  - Retriever HTTP API  (run_retriever.py)    → GET /list_files, POST /v1/retrieve
  - Indexer HTTP API    (run_index_api.py)    → POST/PUT/DELETE/GET /v1/files
"""

import json
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from conftest import SAMPLE_DOCS, FakeSparseEmbedding
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pymilvus import MilvusClient

from mmore.index.indexer import Indexer
from mmore.rag.model import DenseModelConfig, SparseModelConfig
from mmore.run_index_api import (
    _apply_uploaded_file_metadata,
)
from mmore.run_index_api import (
    make_router as make_index_router,
)
from mmore.run_retriever import make_router, save_results
from mmore.type import DocumentMetadata, MultimodalSample

_COLLECTION = "my_docs"


# ---------------------------------------------------------------------------
# Retriever API fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    """Populates a Milvus Lite DB once, shared across the module."""
    path = str(tmp_path_factory.mktemp("retriever_api_db") / "test.db")
    with patch(
        "mmore.index.indexer.SparseModel.from_config",
        return_value=FakeSparseEmbedding(),
    ):
        client = MilvusClient(path, enable_sparse=True)
        indexer = Indexer(
            dense_model_config=DenseModelConfig(model_name="debug"),
            sparse_model_config=SparseModelConfig(
                model_name="naver/splade-cocondenser-selfdistil"
            ),
            client=client,
        )
        indexer.index_documents(SAMPLE_DOCS, collection_name=_COLLECTION)
    return path


@pytest.fixture(scope="module")
def config_path(tmp_path_factory, db_path):
    """Writes RetrieverConfig YAML pointing at the populated DB."""
    cfg = {
        "db": {"uri": db_path, "name": "my_db"},
        "hybrid_search_weight": 0.5,
        "k": 2,
        "collection_name": _COLLECTION,
        "use_web": False,
        "reranker_model_name": None,
    }
    path = tmp_path_factory.mktemp("cfg") / "retriever.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return str(path)


@pytest.fixture(scope="module")
def client(config_path):
    """Builds the FastAPI app via make_router() and returns a TestClient."""
    with patch(
        "mmore.rag.retriever.SparseModel.from_config",
        return_value=FakeSparseEmbedding(),
    ):
        router = make_router(config_path)

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /list_files
# ---------------------------------------------------------------------------


def test_list_files_returns_all_documents(client):
    response = client.get("/list_files", params={"collection_name": _COLLECTION})
    assert response.status_code == 200
    ids = {f["id"] for f in response.json()}
    assert ids == {"doc-1", "doc-2", "doc-3"}


def test_list_files_missing_collection_name(client):
    response = client.get("/list_files")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /v1/retrieve
# ---------------------------------------------------------------------------


def test_retrieve_returns_results(client):
    payload = {
        "fileIds": [],
        "maxMatches": 2,
        "minSimilarity": -1.0,
        "query": "capital of France",
    }
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) <= 2
    assert all("content" in r for r in results)
    assert all("similarity" in r for r in results)
    assert all("fileId" in r for r in results)


def test_retrieve_response_shape(client):
    payload = {
        "fileIds": [],
        "maxMatches": 1,
        "minSimilarity": -1.0,
        "query": "vector database",
    }
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 200
    result = response.json()[0]
    assert "fileId" in result
    assert "content" in result
    assert "similarity" in result


def test_retrieve_with_file_ids_filter(client):
    payload = {
        "fileIds": ["doc-1"],
        "maxMatches": 3,
        "minSimilarity": -1.0,
        "query": "France",
    }
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 200
    results = response.json()
    returned_ids = {r["fileId"] for r in results}
    assert returned_ids <= {"doc-1"}


def test_retrieve_max_matches_respected(client):
    payload = {
        "fileIds": [],
        "maxMatches": 1,
        "minSimilarity": -1.0,
        "query": "anything",
    }
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 200
    assert len(response.json()) <= 1


def test_retrieve_high_min_similarity_returns_empty(client):
    payload = {
        "fileIds": [],
        "maxMatches": 3,
        "minSimilarity": 0.999,
        "query": "France",
    }
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 200
    assert response.json() == []


def test_retrieve_missing_query_returns_422(client):
    payload = {"fileIds": [], "maxMatches": 2, "minSimilarity": -1.0}
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 422


def test_retrieve_missing_max_matches_returns_422(client):
    payload = {"fileIds": [], "minSimilarity": -1.0, "query": "test"}
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 422


def test_retrieve_invalid_max_matches_returns_422(client):
    """maxMatches has ge=1 constraint."""
    payload = {"fileIds": [], "maxMatches": 0, "minSimilarity": -1.0, "query": "test"}
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 422


def test_retrieve_invalid_min_similarity_returns_422(client):
    """minSimilarity must be between -1.0 and 1.0."""
    payload = {
        "fileIds": [],
        "maxMatches": 2,
        "minSimilarity": 5.0,
        "query": "test",
    }
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Utility function: save_results
# ---------------------------------------------------------------------------


def test_save_results_writes_valid_json(tmp_path):
    from langchain_core.documents import Document

    queries = ["What is Paris?"]
    docs = [
        Document(
            page_content="Paris is the capital.",
            metadata=DocumentMetadata(
                file_path="paris.txt",
                extra={
                    "rank": 1,
                    "similarity": 0.9,
                    "id": "1",
                    "page_numbers": [],
                    "paragraph_numbers": [],
                },
            ).to_dict(),
        )
    ]
    results = [docs]
    output_file = tmp_path / "results.json"

    save_results(results, queries, output_file)

    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert len(data) == 1
    assert data[0]["query"] == "What is Paris?"
    assert len(data[0]["context"]) == 1
    assert data[0]["context"][0]["page_content"] == "Paris is the capital."


def test_save_results_multiple_queries(tmp_path):
    from langchain_core.documents import Document

    queries = ["query 1", "query 2"]
    results = [
        [
            Document(
                page_content="doc A",
                metadata=DocumentMetadata(
                    file_path="doc-a.txt",
                    extra={
                        "rank": 1,
                        "similarity": 0.8,
                        "id": "a",
                        "page_numbers": [],
                        "paragraph_numbers": [],
                    },
                ).to_dict(),
            )
        ],
        [
            Document(
                page_content="doc B",
                metadata=DocumentMetadata(
                    file_path="doc-b.txt",
                    extra={
                        "rank": 1,
                        "similarity": 0.7,
                        "id": "b",
                        "page_numbers": [],
                        "paragraph_numbers": [],
                    },
                ).to_dict(),
            )
        ],
    ]
    output_file = tmp_path / "results.json"
    save_results(results, queries, output_file)

    data = json.loads(output_file.read_text())
    assert len(data) == 2
    assert data[0]["query"] == "query 1"
    assert data[1]["query"] == "query 2"


# ---------------------------------------------------------------------------
# Indexer API (run_index_api.py) fixtures
#
# process_files_default is patched to avoid running the full processor pipeline.
# get_indexer is patched to return a Milvus Lite-backed Indexer.
# UPLOAD_DIR is redirected to a tmp directory.
# register_all_processors is patched to skip preloading heavy models.
# ---------------------------------------------------------------------------


def _fake_doc(file_path: str, document_id: str = "doc") -> MultimodalSample:
    """Returns a single fake MultimodalSample with the given file_path in metadata."""
    return MultimodalSample(
        id=f"{document_id}+0",
        document_id=document_id,
        text="Test document content.",
        modalities=[],
        metadata=DocumentMetadata(file_path=file_path),
    )


def _wait_job(tc, job_id: str, timeout: float = 15.0) -> dict:
    """Poll the job status endpoint until the background job is terminal."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = tc.get(f"/v1/jobs/{job_id}")
        if resp.status_code == 200 and resp.json()["status"] in ("done", "failed"):
            return resp.json()
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish in time")


def test_apply_uploaded_file_metadata_preserves_chunk_suffix():
    doc = _fake_doc("/tmp/original-name.txt", document_id="default-doc")
    doc.id = "processor-generated-id+7"

    _apply_uploaded_file_metadata([doc], "client-doc", "original-name.txt")

    assert doc.document_id == "client-doc"
    assert doc.id == "client-doc+7"
    assert doc.metadata.extra["filename"] == "original-name.txt"


@pytest.fixture(scope="module")
def indexer_client(tmp_path_factory):
    """Builds the indexer FastAPI app."""
    upload_dir = str(tmp_path_factory.mktemp("idx_uploads"))
    db_path = str(tmp_path_factory.mktemp("idx_db") / "test.db")
    config_file = str(tmp_path_factory.mktemp("idx_cfg") / "config.yaml")

    cfg = {
        "db": {"uri": db_path, "name": "my_db"},
        "hybrid_search_weight": 0.5,
        "k": 2,
        "collection_name": _COLLECTION,
        "use_web": False,
        "reranker_model_name": None,
    }
    with open(config_file, "w") as f:
        yaml.dump(cfg, f)

    # Create the Milvus Lite indexer
    with patch(
        "mmore.index.indexer.SparseModel.from_config",
        return_value=FakeSparseEmbedding(),
    ):
        milvus_client = MilvusClient(db_path, enable_sparse=True)
        the_indexer = Indexer(
            dense_model_config=DenseModelConfig(model_name="debug"),
            sparse_model_config=SparseModelConfig(
                model_name="naver/splade-cocondenser-selfdistil"
            ),
            client=milvus_client,
        )

    # Pre-create a file for delete/download tests
    pre_id = "pre-existing-doc"
    pre_file = Path(upload_dir) / pre_id
    pre_file.write_bytes(b"Pre-existing file content.")
    the_indexer.index_documents(
        [_fake_doc(str(pre_file), document_id=pre_id)],
        collection_name=_COLLECTION,
    )

    stack = ExitStack()
    stack.enter_context(patch("mmore.run_index_api.UPLOAD_DIR", upload_dir))
    stack.enter_context(patch("mmore.run_index_api.register_all_processors"))
    stack.enter_context(
        patch("mmore.run_index_api.get_indexer", return_value=the_indexer)
    )

    router = make_index_router(config_file)
    app = FastAPI()
    app.include_router(router)
    tc = TestClient(app, raise_server_exceptions=False)

    yield tc, upload_dir, pre_id

    stack.close()


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


def test_index_root(indexer_client):
    tc, *_ = indexer_client
    response = tc.get("/")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /v1/files
# ---------------------------------------------------------------------------


def test_upload_file_success(indexer_client):
    tc, upload_dir, _ = indexer_client
    fake_path = str(Path(upload_dir) / "new-doc.txt")

    with patch(
        "mmore.run_index_api._process_files",
        return_value=[_fake_doc(fake_path)],
    ):
        response = tc.post(
            "/v1/files",
            data={"fileId": "new-doc"},
            files={"file": ("new-doc.txt", b"Hello world", "text/plain")},
        )
        assert response.status_code == 202
        job_id = response.json()["jobId"]
        job = _wait_job(tc, job_id)

    assert job["status"] == "done"
    assert Path(upload_dir, "new-doc").exists()


def test_uploaded_file_has_filename_in_list_files(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    db_path = str(tmp_path / "uploaded_list_files.db")
    config_file = tmp_path / "config.yaml"
    cfg = {
        "db": {"uri": db_path, "name": "my_db"},
        "hybrid_search_weight": 0.5,
        "k": 2,
        "collection_name": _COLLECTION,
        "use_web": False,
        "reranker_model_name": None,
    }
    with open(config_file, "w") as f:
        yaml.dump(cfg, f)

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "mmore.index.indexer.SparseModel.from_config",
                return_value=FakeSparseEmbedding(),
            )
        )
        milvus_client = MilvusClient(db_path, enable_sparse=True)
        the_indexer = Indexer(
            dense_model_config=DenseModelConfig(model_name="debug"),
            sparse_model_config=SparseModelConfig(
                model_name="naver/splade-cocondenser-selfdistil"
            ),
            client=milvus_client,
        )
        stack.enter_context(patch("mmore.run_index_api.UPLOAD_DIR", str(upload_dir)))
        stack.enter_context(patch("mmore.run_index_api.register_all_processors"))
        stack.enter_context(
            patch("mmore.run_index_api.get_indexer", return_value=the_indexer)
        )

        index_app = FastAPI()
        index_app.include_router(make_index_router(str(config_file)))
        index_client = TestClient(index_app, raise_server_exceptions=False)

        uploaded_path = str(upload_dir / "listed-doc.txt")
        stack.enter_context(
            patch(
                "mmore.run_index_api._process_files",
                return_value=[_fake_doc(uploaded_path)],
            )
        )
        response = index_client.post(
            "/v1/files",
            data={"fileId": "listed-doc"},
            files={"file": ("listed-doc.txt", b"Hello list files", "text/plain")},
        )
        assert response.status_code == 202
        assert _wait_job(index_client, response.json()["jobId"])["status"] == "done"

        stack.enter_context(
            patch(
                "mmore.rag.retriever.SparseModel.from_config",
                return_value=FakeSparseEmbedding(),
            )
        )
        retriever_app = FastAPI()
        retriever_app.include_router(make_router(str(config_file)))
        retriever_client = TestClient(retriever_app)

        response = retriever_client.get(
            "/list_files", params={"collection_name": _COLLECTION}
        )

    assert response.status_code == 200
    files_by_id = {file["id"]: file["filename"] for file in response.json()}
    assert files_by_id["listed-doc"] == "listed-doc.txt"
    assert files_by_id["listed-doc"] != "Unknown"


def test_upload_duplicate_file_returns_400(indexer_client):
    tc, upload_dir, _ = indexer_client
    duplicate_id = "duplicate-doc"
    Path(upload_dir, duplicate_id).write_bytes(b"Existing content.")
    with patch("mmore.run_index_api._process_files", return_value=[]):
        response = tc.post(
            "/v1/files",
            data={"fileId": duplicate_id},
            files={"file": ("duplicate-doc.txt", b"Hello again", "text/plain")},
        )
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


def test_upload_failed_processing_does_not_consume_id(indexer_client):
    tc, upload_dir, _ = indexer_client
    file_id = "id"

    # Processing now fails in the background, the upload still returns 202
    with patch("mmore.run_index_api._process_files", side_effect=KeyError("xyz")):
        response = tc.post(
            "/v1/files",
            data={"fileId": file_id},
            files={"file": ("file.xyz", b"bad", "application/octet-stream")},
        )
        assert response.status_code == 202
        job = _wait_job(tc, response.json()["jobId"])
    assert job["status"] == "failed"
    assert not Path(upload_dir, file_id).exists()

    # A failed job releases the id, so it can be reused
    fake_path = str(Path(upload_dir) / "good.txt")
    with patch(
        "mmore.run_index_api._process_files",
        return_value=[_fake_doc(fake_path, file_id)],
    ):
        response = tc.post(
            "/v1/files",
            data={"fileId": file_id},
            files={"file": ("file.txt", b"good", "text/plain")},
        )
        assert response.status_code == 202
        assert _wait_job(tc, response.json()["jobId"])["status"] == "done"
    assert Path(upload_dir, file_id).read_bytes() == b"good"


# ---------------------------------------------------------------------------
# POST /v1/files/bulk
# ---------------------------------------------------------------------------


def test_upload_bulk_files_success(indexer_client):
    tc, *_ = indexer_client

    def fake_process(pool, input_dir, collection_name, extensions, device, output_path):
        # Each job processes a single file
        path = next(p for p in Path(input_dir).iterdir() if p.is_file())
        count = 2 if path.name.startswith("bulk-1") else 1
        return [_fake_doc(str(path)) for _ in range(count)]

    with patch(
        "mmore.run_index_api._process_files",
        side_effect=fake_process,
    ):
        response = tc.post(
            "/v1/files/bulk",
            data={"listIds": "bulk-1,bulk-2"},
            files=[
                ("files", ("bulk-1.txt", b"Bulk content 1", "text/plain")),
                ("files", ("bulk-2.txt", b"Bulk content 2", "text/plain")),
            ],
        )
        assert response.status_code == 202
        jobs_by_id = {j["fileId"]: j for j in response.json()["jobs"]}
        assert set(jobs_by_id) == {"bulk-1", "bulk-2"}
        results = {fid: _wait_job(tc, job["jobId"]) for fid, job in jobs_by_id.items()}

    assert results["bulk-1"]["status"] == "done"
    assert results["bulk-1"]["result"]["chunks"] == 2
    assert results["bulk-2"]["status"] == "done"
    assert results["bulk-2"]["result"]["chunks"] == 1


def test_upload_bulk_files_allows_duplicate_uploaded_filenames(indexer_client):
    tc, upload_dir, _ = indexer_client

    def fake_process(pool, input_dir, collection_name, extensions, device, output_path):
        path = next(p for p in Path(input_dir).iterdir() if p.is_file())
        return [_fake_doc(str(path))]

    with patch(
        "mmore.run_index_api._process_files",
        side_effect=fake_process,
    ):
        response = tc.post(
            "/v1/files/bulk",
            data={"listIds": "same-name-1,same-name-2"},
            files=[
                ("files", ("same.txt", b"First content", "text/plain")),
                ("files", ("same.txt", b"Second content", "text/plain")),
            ],
        )
        assert response.status_code == 202
        jobs_by_id = {j["fileId"]: j for j in response.json()["jobs"]}
        assert set(jobs_by_id) == {"same-name-1", "same-name-2"}
        for job in jobs_by_id.values():
            assert _wait_job(tc, job["jobId"])["status"] == "done"

    assert Path(upload_dir, "same-name-1").exists()
    assert Path(upload_dir, "same-name-2").exists()


def test_upload_bulk_mismatched_ids_returns_400(indexer_client):
    tc, *_ = indexer_client
    with patch("mmore.run_index_api._process_files", return_value=[]):
        response = tc.post(
            "/v1/files/bulk",
            data={"listIds": "only-one-id"},
            files=[
                ("files", ("a.txt", b"Content A", "text/plain")),
                ("files", ("b.txt", b"Content B", "text/plain")),
            ],
        )
    assert response.status_code == 400
    assert "doesn't match" in response.json()["detail"]


def test_upload_bulk_failed_processing_does_not_consume_ids(indexer_client):
    tc, upload_dir, _ = indexer_client
    ids = ["id-1", "id-2"]

    with patch("mmore.run_index_api._process_files", side_effect=KeyError("xyz")):
        response = tc.post(
            "/v1/files/bulk",
            data={"listIds": ",".join(ids)},
            files=[
                ("files", ("file.xyz", b"bad A", "application/octet-stream")),
                ("files", ("file.xyz", b"bad B", "application/octet-stream")),
            ],
        )
        assert response.status_code == 202
        for job in response.json()["jobs"]:
            assert _wait_job(tc, job["jobId"])["status"] == "failed"
    for file_id in ids:
        assert not Path(upload_dir, file_id).exists()

    def fake_process(pool, input_dir, collection_name, extensions, device, output_path):
        path = next(p for p in Path(input_dir).iterdir() if p.is_file())
        return [_fake_doc(str(path))]

    with patch(
        "mmore.run_index_api._process_files",
        side_effect=fake_process,
    ):
        response = tc.post(
            "/v1/files/bulk",
            data={"listIds": ",".join(ids)},
            files=[
                ("files", ("file.txt", b"good A", "text/plain")),
                ("files", ("file.txt", b"good B", "text/plain")),
            ],
        )
        assert response.status_code == 202
        for job in response.json()["jobs"]:
            assert _wait_job(tc, job["jobId"])["status"] == "done"
    assert Path(upload_dir, ids[0]).read_bytes() == b"good A"
    assert Path(upload_dir, ids[1]).read_bytes() == b"good B"


# ---------------------------------------------------------------------------
# PUT /v1/files/{fileId}
# ---------------------------------------------------------------------------


def test_update_existing_file_success(indexer_client):
    tc, upload_dir, _ = indexer_client
    update_id = "update-doc"
    Path(upload_dir, update_id).write_bytes(b"Original content.")

    fake_path = str(Path(upload_dir) / "update-doc.txt")
    with patch(
        "mmore.run_index_api._process_files",
        return_value=[_fake_doc(fake_path, update_id)],
    ):
        response = tc.put(
            f"/v1/files/{update_id}",
            files={"file": ("update-doc.txt", b"Updated content.", "text/plain")},
        )
        assert response.status_code == 202
        assert response.json()["fileId"] == update_id
        assert _wait_job(tc, response.json()["jobId"])["status"] == "done"

    assert Path(upload_dir, update_id).read_bytes() == b"Updated content."


def test_update_nonexistent_file_returns_404(indexer_client):
    tc, *_ = indexer_client
    with patch("mmore.run_index_api._process_files", return_value=[]):
        response = tc.put(
            "/v1/files/does-not-exist",
            files={"file": ("x.txt", b"content", "text/plain")},
        )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /v1/files/{fileId}
# ---------------------------------------------------------------------------


def test_delete_existing_file_success(indexer_client):
    tc, upload_dir, pre_id = indexer_client
    response = tc.delete(f"/v1/files/{pre_id}")
    assert response.status_code == 200
    assert not Path(upload_dir, pre_id).exists()


def test_delete_nonexistent_file_returns_404(indexer_client):
    tc, *_ = indexer_client
    response = tc.delete("/v1/files/does-not-exist")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /v1/files/{fileId}
# ---------------------------------------------------------------------------


def test_download_existing_file_success(indexer_client):
    tc, upload_dir, _ = indexer_client
    Path(upload_dir, "new-doc").write_bytes(b"Downloadable content.")
    response = tc.get("/v1/files/new-doc")
    assert response.status_code == 200


def test_download_nonexistent_file_returns_404(indexer_client):
    tc, *_ = indexer_client
    response = tc.get("/v1/files/does-not-exist")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /v1/jobs/{jobId} and /v1/jobs/{jobId}/events
# ---------------------------------------------------------------------------


def test_get_unknown_job_returns_404(indexer_client):
    tc, *_ = indexer_client
    response = tc.get("/v1/jobs/does-not-exist")
    assert response.status_code == 404


def test_job_events_stream_reaches_done(indexer_client):
    tc, upload_dir, _ = indexer_client
    fake_path = str(Path(upload_dir) / "sse-doc.txt")
    with patch(
        "mmore.run_index_api._process_files",
        return_value=[_fake_doc(fake_path)],
    ):
        response = tc.post(
            "/v1/files",
            data={"fileId": "sse-doc"},
            files={"file": ("sse-doc.txt", b"stream me", "text/plain")},
        )
        job_id = response.json()["jobId"]
        with tc.stream("GET", f"/v1/jobs/{job_id}/events") as stream:
            body = "".join(stream.iter_text())

    # The stream stays open until the job is terminal, so "done" is always sent
    assert "data:" in body
    assert "done" in body
