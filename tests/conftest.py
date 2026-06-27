import importlib.util
import json
from typing import Dict, List

import pytest

from mmore.type import DocumentMetadata, MultimodalSample

# colvision is a conflicting extra installed in its own venv (see tests.yml)
if importlib.util.find_spec("colpali_engine") is None:
    collect_ignore = ["test_colvision.py"]

try:
    from langchain_milvus.utils.sparse import BaseSparseEmbedding

    class FakeSparseEmbedding(BaseSparseEmbedding):
        """Fake sparse embedder for test purposes."""

        def embed_query(self, query: str) -> Dict[int, float]:
            return {0: 1.0, 1: float(len(query))}

        def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
            return [{0: 1.0, i + 1: float(len(t))} for i, t in enumerate(texts)]

except ImportError:
    # langchain_milvus is part of the optional `index`/`rag` extras.
    # Tests that need FakeSparseEmbedding should be marked and will be
    # collected only when those extras are installed.
    FakeSparseEmbedding = None  # type: ignore[assignment,misc]


SAMPLE_DOCS = [
    MultimodalSample(
        id="doc-1",
        document_id="doc-1",
        text="Paris is the capital of France.",
        modalities=[],
        metadata=DocumentMetadata(),
    ),
    MultimodalSample(
        id="doc-2",
        document_id="doc-2",
        text="The Eiffel Tower stands 330 metres tall.",
        modalities=[],
        metadata=DocumentMetadata(extra={"author": "Alice"}),
    ),
    MultimodalSample(
        id="doc-3",
        document_id="doc-3",
        text="Milvus is an open-source vector database.",
        modalities=[],
        metadata=DocumentMetadata(),
    ),
]


@pytest.fixture
def make_sample():
    def _make(file_path: str, text: str = "x", **metadata) -> MultimodalSample:
        return MultimodalSample.from_dict(
            {
                "text": text,
                "modalities": [],
                "metadata": {"file_path": file_path, **metadata},
            }
        )

    return _make


@pytest.fixture
def write_jsonl():
    def _write(path: str, samples: list[MultimodalSample]) -> None:
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict()) + "\n")

    return _write


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run tests that require a GPU",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="Pass --gpu to run GPU tests")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
