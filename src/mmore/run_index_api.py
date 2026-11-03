import argparse
import asyncio
import json
import logging
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from pathlib import Path as FilePath
from typing import Any, Callable, Dict, List, Optional

import torch
import uvicorn
from fastapi import (
    APIRouter,
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse, StreamingResponse

logger = logging.getLogger(__name__)
RETRIVER_EMOJI = "🗂️"
logging.basicConfig(
    format=f"[INDEX API {RETRIVER_EMOJI} -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

from mmore.profiler import enable_profiling_from_env

from .job_queue import DuplicateJobError, Job, JobQueue, QueueFullError
from .process.processors import register_all_processors
from .rag.retriever import RetrieverConfig
from .type import MultimodalSample
from .utils import get_indexer, load_config, process_files_default

UPLOAD_DIR: str = "./uploads"

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


POLL_INTERVAL = 2.0
HEARTBEAT_SECONDS = 15

# Rows per Milvus iterator page when reading every chunk of a file (vectors included)
FILE_ROWS_BATCH_SIZE = 1000

# Columns of the collection schema, not metadata (see Indexer._create_collection_with_schema)
SCHEMA_FIELDS = frozenset(
    {"id", "document_id", "text", "dense_embedding", "sparse_embedding"}
)

# Metadata managed by the API or the processing pipeline, never patchable
PROTECTED_METADATA = SCHEMA_FIELDS | {"file_path", "processed_at", "processor_type"}

PATCHABLE_TYPES = (str, int, float, bool)


def _process_files(pool, input_dir, collection_name, extensions, device, output_path):
    """Run processing in the device's subprocess."""
    return pool.submit(
        process_files_default,
        input_dir,
        collection_name,
        extensions,
        device=device,
        output_path=output_path,
    ).result()


def _job_payload(job: Job) -> dict:
    return {
        "jobId": job.id,
        "fileId": job.file_id,
        "filename": job.filename,
        "status": job.status.value,
        "device": job.device,
        "result": job.result,
        "error": job.error,
    }


def _apply_uploaded_file_metadata(
    documents: List[MultimodalSample], file_id: str, filename: str
) -> None:
    """Bind processed chunks to the API file ID and persist the original filename."""
    for doc in documents:
        chunk_id = doc.id.rsplit("+")[1] if "+" in doc.id else None
        doc.document_id = file_id
        doc.id = f"{file_id}+{chunk_id}" if chunk_id else file_id

        doc.metadata.extra["filename"] = filename
        doc.metadata.file_path = str(FilePath(UPLOAD_DIR) / file_id)


def _file_level_metadata(rows: List[dict]) -> dict:
    """Return the metadata shared by every chunk of a file.

    A key is file-level only if it is present on all rows with the same value,
    chunk-level keys (e.g. paragraph_positions) are left out.
    """
    if not rows:
        return {}

    # Candidates: every metadata of the first chunk
    shared = {k: v for k, v in rows[0].items() if k not in SCHEMA_FIELDS}

    # Drop the keys missing or with another value on any other chunk
    for row in rows[1:]:
        shared = {k: v for k, v in shared.items() if k in row and row[k] == v}
    return shared


def _json_type(value: Any) -> str:
    """JSON type name of a scalar, bool checked before int since bool subclasses int."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _validate_metadata_patch(patch: Dict[str, Any], rows: List[dict]) -> List[str]:
    """Check a merge patch against the file's chunks, return the list of errors.

    The patch is a merge: a new key is added, an existing key is updated.
    - protected keys cannot be patched
    - an existing key must be file-level (same value on all chunks)
    - an existing key keeps its JSON type
    """
    shared = _file_level_metadata(rows)
    existing = {k for row in rows for k in row} - SCHEMA_FIELDS
    errors = []
    for key, value in patch.items():
        if key in PROTECTED_METADATA:
            errors.append(f"'{key}' is protected and cannot be patched")
        elif not isinstance(value, PATCHABLE_TYPES):
            errors.append(
                f"'{key}' must be a string, number or boolean, got {_json_type(value)}"
            )
        elif key in existing and key not in shared:
            errors.append(
                f"'{key}' differs across the chunks of the file, "
                "only file-level metadata can be patched"
            )
        elif (
            key in shared
            and shared[key] is not None
            and _json_type(value) != _json_type(shared[key])
        ):
            errors.append(
                f"'{key}' must stay a {_json_type(shared[key])}, "
                f"got {_json_type(value)}"
            )
    return errors


def make_router(config_path: str) -> APIRouter:
    router = APIRouter()

    config = load_config(config_path, RetrieverConfig)

    MILVUS_URI = config.db.uri or "./proc_demo.db"
    MILVUS_DB = config.db.name or "my_db"
    COLLECTION_NAME = config.collection_name or "my_docs"

    # Initialize the index database and the processors
    indexer = get_indexer(COLLECTION_NAME, MILVUS_URI, MILVUS_DB)
    milvus_client = indexer.client
    register_all_processors(preload=True)

    jobs = JobQueue(
        jobs_per_gpu=config.jobs_per_gpu,
        max_queue_size=config.max_queue_size,
    )

    mp_context = multiprocessing.get_context("spawn")
    process_pools = {
        device: ProcessPoolExecutor(
            max_workers=config.jobs_per_gpu, mp_context=mp_context
        )
        for device in jobs.devices
    }

    def _shutdown():
        jobs.shutdown()
        for pool in process_pools.values():
            pool.shutdown(wait=True)

    router.add_event_handler("shutdown", _shutdown)

    def _existing_collection(collection_name: Optional[str]) -> str:
        """Resolve the requested collection, raise HTTPException 404 if it does not exist."""
        if collection_name is None:
            collection_name = COLLECTION_NAME
        if not milvus_client.has_collection(collection_name):
            raise HTTPException(
                status_code=404, detail=f"Collection {collection_name} not found"
            )
        return collection_name

    def _existing_indexed_file(
        file_id: str, collection_name: str, output_fields: Optional[list[str]] = None
    ) -> dict:
        """Return one indexed row of the file, raise HTTPException 404 if the file is not in the collection."""
        rows = milvus_client.query(
            collection_name=collection_name,
            filter=f"document_id == {json.dumps(file_id)}",
            output_fields=output_fields or ["document_id"],
            limit=1,
        )
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"File with ID {file_id} not found in collection {collection_name}",
            )
        return rows[0]

    def _existing_file_rows(file_id: str, collection_name: str) -> List[dict]:
        """Return every chunk of a corresponding file, raise HTTPException 404 if it has none."""
        iterator = milvus_client.query_iterator(
            collection_name=collection_name,
            filter=f"document_id == {json.dumps(file_id)}",
            output_fields=["*"],
            batch_size=FILE_ROWS_BATCH_SIZE,
        )
        try:
            rows = list(chain.from_iterable(iter(iterator.next, [])))
        finally:
            iterator.close()
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"File with ID {file_id} not found in collection {collection_name}",
            )
        return rows

    def _stage_upload(file: UploadFile, filename: str) -> tuple[str, str]:
        """Save the uploaded bytes now, while the request is alive.

        Returns (job_dir, input_dir). The worker removes job_dir when done.
        """
        job_dir = tempfile.mkdtemp(prefix="index_job_")
        input_dir = os.path.join(job_dir, "input")
        os.makedirs(input_dir)
        target = os.path.realpath(os.path.join(input_dir, os.path.basename(filename)))
        if os.path.dirname(target) != os.path.realpath(input_dir):
            raise HTTPException(status_code=422, detail="Invalid filename")
        with open(target, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return job_dir, input_dir

    def _make_ingest_job(
        job_dir: str,
        input_dir: str,
        collection_name: str,
        file_id: str,
        filename: str,
        replace: bool,
    ) -> Callable[[str], dict]:
        extension = FilePath(filename).suffix.lower()

        def ingest(device: str) -> dict:
            try:
                # Processing runs in the device's subprocess
                documents = _process_files(
                    process_pools[device],
                    input_dir,
                    collection_name,
                    [extension],
                    device,
                    os.path.join(job_dir, "out"),
                )
                _apply_uploaded_file_metadata(documents, file_id, filename)

                indexer = get_indexer(
                    collection_name, MILVUS_URI, MILVUS_DB, device=device
                )
                if str(device).startswith("cuda"):
                    torch.cuda.set_device(torch.device(device))
                if replace:
                    indexer.client.delete(
                        collection_name=collection_name,
                        filter=f"document_id == {json.dumps(file_id)}",
                    )
                indexer.index_documents(
                    documents=documents, collection_name=collection_name
                )
                indexer.client.flush(collection_name)

                # Persist the permanent copy only on success
                dest = FilePath(UPLOAD_DIR) / file_id
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(os.path.join(input_dir, filename), dest)
                return {"chunks": len(documents)}
            finally:
                shutil.rmtree(job_dir, ignore_errors=True)

        return ingest

    @router.get("/", tags=["Health"], summary="Health check")
    async def root():
        return {
            "message": "Indexer API is running (what are you doing here...go catch it!)"
        }

    # SINGLE FILE UPLOAD ENDPOINT
    @router.post(
        "/v1/files",
        status_code=202,
        tags=["File Operations"],
        summary="Upload a single file",
        responses={
            202: {
                "description": "Job queued",
                "content": {
                    "application/json": {
                        "example": {"jobId": "a1b2c3d4...", "fileId": "example123"}
                    }
                },
            },
            409: {
                "description": "File ID already exists or is already being processed"
            },
            422: {"description": "Uploaded file has no filename"},
            503: {"description": "Job queue is full, retry later"},
        },
    )
    async def upload_file(
        fileId: str = Form(..., description="Unique identifier for the file"),
        file: UploadFile = File(..., description="The file content"),
        collectionName: Optional[str] = Form(
            None,
            description="Collection to index into (defaults to the "
            "configured collection)",
        ),
    ):
        """
        Queue a new file for processing and indexing.

        Returns a jobId immediately, the work runs in the background.
        """
        if file.filename is None:
            raise HTTPException(
                status_code=422, detail="Provided file should have a filename"
            )
        # Indexing creates the collection if it does not exist yet
        collection_name = collectionName or COLLECTION_NAME
        if (FilePath(UPLOAD_DIR) / fileId).exists():
            raise HTTPException(
                status_code=409, detail=f"File with ID {fileId} already exists"
            )

        job_dir, input_dir = _stage_upload(file, file.filename)
        await file.close()
        ingest = _make_ingest_job(
            job_dir,
            input_dir,
            collection_name=collection_name,
            file_id=fileId,
            filename=file.filename,
            replace=False,
        )
        try:
            job_id = jobs.submit(fileId, file.filename, ingest)
        except DuplicateJobError:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(
                status_code=409,
                detail=f"File with ID {fileId} is already being processed",
            )
        except QueueFullError:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=503, detail="Server busy, retry later")

        return {"jobId": job_id, "fileId": fileId}

    @router.post(
        "/v1/files/bulk",
        status_code=202,
        tags=["File Operations"],
        summary="Upload multiple files with IDs",
        responses={
            202: {
                "description": "Per-file outcome (jobId or error)",
                "content": {
                    "application/json": {
                        "example": {
                            "jobs": [
                                {"fileId": "doc1", "jobId": "a1b2c3..."},
                                {"fileId": "doc2", "error": "already exists"},
                            ]
                        }
                    }
                },
            },
            400: {"description": "Number of IDs does not match number of files"},
        },
    )
    async def upload_files(
        listIds: List[str] = Form(..., description="List of IDs for the files"),
        files: List[UploadFile] = File(..., description="Files to upload"),
        collectionName: Optional[str] = Form(
            None,
            description="Collection to index into (defaults to the "
            "configured collection)",
        ),
    ):
        """
        Queue multiple files, one independent job per file.

        Returns a per-file outcome (jobId or error), so one bad file does not
        fail the whole batch.
        """
        # Indexing creates the collection if it does not exist yet
        collection_name = collectionName or COLLECTION_NAME
        listIds = [
            file_id.strip()
            for ids in listIds
            for file_id in ids.split(",")
            if file_id.strip()
        ]
        if len(listIds) != len(files):
            raise HTTPException(
                status_code=400,
                detail=f"Number of IDs ({len(listIds)}) doesn't match number of files ({len(files)})",
            )

        results = []
        for file, file_id in zip(files, listIds):
            if file.filename is None:
                await file.close()
                results.append({"fileId": file_id, "error": "missing filename"})
                continue
            if (FilePath(UPLOAD_DIR) / file_id).exists():
                await file.close()
                results.append({"fileId": file_id, "error": "already exists"})
                continue

            job_dir, input_dir = _stage_upload(file, file.filename)
            await file.close()
            ingest = _make_ingest_job(
                job_dir,
                input_dir,
                collection_name=collection_name,
                file_id=file_id,
                filename=file.filename,
                replace=False,
            )
            try:
                job_id = jobs.submit(file_id, file.filename, ingest)
                results.append({"fileId": file_id, "jobId": job_id})
            except DuplicateJobError:
                shutil.rmtree(job_dir, ignore_errors=True)
                results.append({"fileId": file_id, "error": "already being processed"})
            except QueueFullError:
                shutil.rmtree(job_dir, ignore_errors=True)
                results.append({"fileId": file_id, "error": "queue full"})

        return {"jobs": results}

    @router.put(
        "/v1/files/{fileId}",
        status_code=202,
        tags=["File Operations"],
        summary="Replace an existing file and re-index",
        responses={
            202: {
                "description": "Replacement job queued",
                "content": {
                    "application/json": {
                        "example": {"jobId": "a1b2c3d4...", "fileId": "doc123"}
                    }
                },
            },
            404: {"description": "File or collection not found"},
            409: {"description": "File is already being processed"},
            422: {"description": "Uploaded file has no filename"},
            503: {"description": "Job queue is full, retry later"},
        },
    )
    async def update_file(
        fileId: str = Path(..., description="ID of the file to update"),
        file: UploadFile = File(..., description="The new file content"),
        collectionName: Optional[str] = Form(
            None,
            description="Collection the file is indexed in (defaults to the "
            "configured collection)",
        ),
    ):
        """
        Queue a replacement for an existing file and re-index it.

        The old vectors are deleted and the new ones inserted only after
        processing succeeds, so a failure leaves the existing document intact.
        """
        collection_name = _existing_collection(collectionName)
        if not (FilePath(UPLOAD_DIR) / fileId).exists():
            raise HTTPException(
                status_code=404, detail=f"File with ID {fileId} not found"
            )
        _existing_indexed_file(fileId, collection_name)
        if file.filename is None:
            raise HTTPException(
                status_code=422, detail="Provided file should have a filename"
            )

        job_dir, input_dir = _stage_upload(file, file.filename)
        await file.close()
        ingest = _make_ingest_job(
            job_dir,
            input_dir,
            collection_name=collection_name,
            file_id=fileId,
            filename=file.filename,
            replace=True,
        )
        try:
            job_id = jobs.submit(fileId, file.filename, ingest)
        except DuplicateJobError:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(
                status_code=409,
                detail=f"File with ID {fileId} is already being processed",
            )
        except QueueFullError:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=503, detail="Server busy, retry later")

        return {"jobId": job_id, "fileId": fileId}

    @router.delete(
        "/v1/files/{fileId}",
        tags=["File Operations"],
        summary="Delete a file and remove its vector entry",
        responses={
            200: {
                "description": "File deleted",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "success",
                            "message": "File successfully deleted",
                            "fileId": "doc123",
                        }
                    }
                },
            },
            404: {"description": "File or collection not found"},
            500: {"description": "Internal error while deleting the file"},
        },
    )
    async def delete_file(
        fileId: str = Path(..., description="ID of the file to delete"),
        collectionName: Optional[str] = Query(
            None,
            description="Collection the file is indexed in (defaults to the "
            "configured collection)",
        ),
    ):
        """
        Delete a file from the system.

        """
        try:
            collection_name = _existing_collection(collectionName)

            # Check if file exists
            file_storage_path = FilePath(UPLOAD_DIR) / fileId
            if not file_storage_path.exists():
                raise HTTPException(
                    status_code=404, detail=f"File with ID {fileId} not found"
                )
            _existing_indexed_file(fileId, collection_name)

            # Delete from vector database first. Ensure that a failure keeps the local copy
            delete_result = milvus_client.delete(
                collection_name=collection_name,
                filter=f"document_id == {json.dumps(fileId)}",
            )
            logger.info(f"Deleted document from vector DB: {delete_result}")

            # Delete the physical file
            os.remove(file_storage_path)

            return {
                "status": "success",
                "message": "File successfully deleted",
                "fileId": fileId,
            }

        except HTTPException as e:
            raise e

        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/v1/files/{fileId}",
        tags=["File Operations"],
        summary="Download a file by its ID",
        response_class=FileResponse,
        responses={
            200: {
                "description": "Binary file content",
                "content": {"application/octet-stream": {}},
            },
            404: {"description": "File or collection not found"},
            500: {"description": "Internal error while retrieving the file"},
        },
    )
    async def download_file(
        fileId: str = Path(..., description="ID of the file to download"),
        collectionName: Optional[str] = Query(
            None,
            description="Collection the file is indexed in (defaults to the "
            "configured collection)",
        ),
    ):
        """
        Download a file from the system.
        """
        try:
            collection_name = _existing_collection(collectionName)

            # Check if file exists
            file_storage_path = FilePath(UPLOAD_DIR) / fileId
            if not file_storage_path.exists():
                raise HTTPException(
                    status_code=404, detail=f"File with ID {fileId} not found"
                )

            # Retrieve the filename from metadata, it is file-level so any chunk has it.
            # Fall back to the stored path for files indexed before it was persisted
            row = _existing_indexed_file(
                fileId, collection_name, ["file_path", "filename"]
            )
            filename = row.get("filename") or row["file_path"].split("/")[-1]

            # Return the file
            return FileResponse(
                path=file_storage_path,
                filename=filename,
                media_type="application/octet-stream",
            )

        except HTTPException as e:
            raise e

        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/v1/files/{fileId}/metadata",
        tags=["File Operations"],
        summary="Get the file-level metadata of a file",
        responses={
            200: {
                "description": "File-level metadata (shared by every chunk)",
                "content": {
                    "application/json": {
                        "example": {
                            "fileId": "doc123",
                            "collectionName": "my_docs",
                            "chunks": 12,
                            "metadata": {
                                "filename": "report.pdf",
                                "file_path": "./uploads/doc123",
                            },
                        }
                    }
                },
            },
            404: {"description": "File or collection not found"},
        },
    )
    def get_file_metadata(
        fileId: str = Path(..., description="ID of the file"),
        collectionName: Optional[str] = Query(
            None,
            description="Collection the file is indexed in (defaults to the "
            "configured collection)",
        ),
    ):
        """
        Return the metadata shared by every chunk of the file.

        """
        collection_name = _existing_collection(collectionName)
        rows = _existing_file_rows(fileId, collection_name)
        return {
            "fileId": fileId,
            "collectionName": collection_name,
            "chunks": len(rows),
            "metadata": _file_level_metadata(rows),
        }

    @router.patch(
        "/v1/files/{fileId}/metadata",
        tags=["File Operations"],
        summary="Patch the file-level metadata of a file",
        responses={
            200: {
                "description": "Metadata after the patch",
                "content": {
                    "application/json": {
                        "example": {
                            "fileId": "doc123",
                            "collectionName": "my_docs",
                            "chunks": 12,
                            "metadata": {
                                "filename": "renamed.pdf",
                                "file_path": "./uploads/doc123",
                            },
                        }
                    }
                },
            },
            404: {"description": "File or collection not found"},
            422: {
                "description": "Invalid patch (protected key, wrong "
                "type, non-scalar or null value, or chunk-level key)"
            },
        },
    )
    def patch_file_metadata(
        fileId: str = Path(..., description="ID of the file"),
        patch: Dict[str, Any] = Body(
            ...,
            description="Merged into the file's metadata: new keys are added, "
            "existing keys are updated, the others are left untouched",
            examples=[{"filename": "renamed.pdf"}],
        ),
        collectionName: Optional[str] = Query(
            None,
            description="Collection the file is indexed in (defaults to the "
            "configured collection)",
        ),
    ):
        """
        Patch the file-level metadata on every chunk of the file, without re-indexing.

        """
        collection_name = _existing_collection(collectionName)
        if not patch:
            raise HTTPException(status_code=422, detail="Empty patch")

        rows = _existing_file_rows(fileId, collection_name)
        errors = _validate_metadata_patch(patch, rows)
        if errors:
            raise HTTPException(status_code=422, detail=errors)

        # Set the patched keys on every chunk, the others are left untouched
        for row in rows:
            row.update(patch)

        # Full-row upsert: current Milvus Lite version does not support partial updates
        for i in range(0, len(rows), FILE_ROWS_BATCH_SIZE):
            milvus_client.upsert(
                collection_name=collection_name,
                data=rows[i : i + FILE_ROWS_BATCH_SIZE],
            )
        milvus_client.flush(collection_name)

        logger.info(f"Patched metadata of {fileId} ({len(rows)} chunks): {patch}")
        return {
            "fileId": fileId,
            "collectionName": collection_name,
            "chunks": len(rows),
            "metadata": _file_level_metadata(rows),
        }

    @router.patch(
        "/v1/files/{fileId}/chunks/{chunkId}/metadata",
        tags=["File Operations"],
        summary="Patch the metadata of a single chunk",
        responses={
            200: {
                "description": "Chunk metadata after the patch",
                "content": {
                    "application/json": {
                        "example": {
                            "fileId": "doc123",
                            "chunkId": "3",
                            "collectionName": "my_docs",
                            "metadata": {
                                "filename": "report.pdf",
                                "file_path": "./uploads/doc123",
                                "paragraph_positions": [[120, 340]],
                                "reviewed": True,
                            },
                        }
                    }
                },
            },
            404: {"description": "Chunk or collection not found"},
            422: {
                "description": "Invalid patch (protected key, wrong "
                "type, non-scalar or null value)"
            },
        },
    )
    def patch_chunk_metadata(
        fileId: str = Path(..., description="ID of the file"),
        chunkId: str = Path(..., description="ID of the chunk within the file"),
        patch: Dict[str, Any] = Body(
            ...,
            description="Merged into the chunk's metadata: new keys are added, "
            "existing keys are updated, the others are left untouched",
            examples=[{"reviewed": True}],
        ),
        collectionName: Optional[str] = Query(
            None,
            description="Collection the file is indexed in (defaults to the "
            "configured collection)",
        ),
    ):
        """
        Patch the metadata of one chunk only, without re-indexing.

        The other chunks of the file are left untouched: patching a file-level key
        here makes it differ across the chunks, choosing the right key is the caller's
        responsibility.
        """
        collection_name = _existing_collection(collectionName)
        if not patch:
            raise HTTPException(status_code=422, detail="Empty patch")

        rows = milvus_client.query(
            collection_name=collection_name,
            filter=f"id == {json.dumps(f'{fileId}+{chunkId}')}",
            output_fields=["*"],
            limit=1,
        )
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"Chunk {chunkId} of file {fileId} not found "
                f"in collection {collection_name}",
            )
        row = rows[0]
        # A single chunk: only the protected keys, value and type checks apply
        errors = _validate_metadata_patch(patch, [row])
        if errors:
            raise HTTPException(status_code=422, detail=errors)

        row.update(patch)
        milvus_client.upsert(collection_name=collection_name, data=[row])
        milvus_client.flush(collection_name)

        logger.info(f"Patched metadata of chunk {fileId}+{chunkId}: {patch}")
        return {
            "fileId": fileId,
            "chunkId": chunkId,
            "collectionName": collection_name,
            "metadata": {k: v for k, v in row.items() if k not in SCHEMA_FIELDS},
        }

    @router.get(
        "/v1/jobs/{jobId}",
        tags=["Jobs"],
        summary="Get a one-shot job status snapshot",
        responses={
            200: {
                "description": "Job status snapshot",
                "content": {
                    "application/json": {
                        "example": {
                            "jobId": "a1b2c3...",
                            "fileId": "doc1",
                            "filename": "doc.pdf",
                            "status": "done",
                            "device": "cuda:0",
                            "result": {"chunks": 12},
                            "error": None,
                        }
                    }
                },
            },
            404: {"description": "Unknown or expired job"},
        },
    )
    async def get_job(jobId: str = Path(..., description="ID of the job")):
        """One-shot job status snapshot, fallback for when the SSE stream drops."""
        job = jobs.get(jobId)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job {jobId}")
        return _job_payload(job)

    @router.get(
        "/v1/jobs/{jobId}/events",
        tags=["Jobs"],
        summary="Stream job status updates over SSE",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": "Server-Sent Events stream of job status changes, "
                "closed when the job is done or failed",
                "content": {"text/event-stream": {}},
            },
        },
    )
    async def stream_job_events(jobId: str = Path(..., description="ID of the job")):
        """Push job status updates to the client over SSE until the job ends."""

        async def event_stream():
            last: Optional[str] = None
            idle = 0.0
            while True:
                job = jobs.get(jobId)
                status = job.status.value if job else "unknown"
                if status != last:
                    last = status
                    idle = 0.0
                    payload = (
                        _job_payload(job) if job else {"jobId": jobId, "status": status}
                    )
                    yield f"data: {json.dumps(payload)}\n\n"
                    if job is None or job.status.is_terminal:
                        return
                else:
                    idle += POLL_INTERVAL
                    if idle >= HEARTBEAT_SECONDS:
                        idle = 0.0
                        yield ": keepalive\n\n"
                await asyncio.sleep(POLL_INTERVAL)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router


def run_api(config_file: str, host: str, port: int):
    router = make_router(config_file)

    app = FastAPI(title="Indexer API")
    app.include_router(router)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    enable_profiling_from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", required=True, help="Path to the retriever configuration file."
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host on which the API should be run."
    )
    parser.add_argument(
        "--port", default=8000, help="Port on which the API should be run."
    )
    args = parser.parse_args()

    run_api(args.config_file, args.host, args.port)
