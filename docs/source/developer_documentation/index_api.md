# 🔌 Indexer API Documentation

```{image} ../doc_images/index_api.png
:width: 900px
:align: center
:alt: Indexer API illustration
```

## Overview

The **Indexer API** allows users to **upload, update, download, delete, and index documents** into a Milvus vector database for retrieval-augmented generation (RAG) and search applications.

Uploads are **asynchronous**: the upload endpoints validate the request, queue a background job, and return `202 Accepted` with a `jobId` immediately. Processing and indexing then run in the background, one job per GPU. Clients track progress with the job status endpoints (snapshot or SSE stream).

Every file operation endpoint takes an optional `collectionName` to run the operation on a collection other than the configured `collection_name`. If the passed collection does not exist, uploads create it, while update, delete, download and metadata operations return `404`.

## ⚙️ Backend server setup

### Setup Instructions

#### 1. Configure the server

The server reads everything from the YAML file passed with `--config-file` (a
`RetrieverConfig`). The relevant fields, with their defaults:

```yaml
db:
  uri: ./proc_demo.db   # Milvus Lite file, or a Standalone server e.g. http://localhost:19530
  name: my_db
collection_name: my_docs
jobs_per_gpu: 1         # upload jobs processed per GPU at once
max_queue_size: null    # pending-job cap
```

#### 2. Run the server

To start the server, run this command:

```bash
python3 -m mmore index-api --config-file /path/to/config.yaml --host the_host --port the_port
```

This command:

- starts the Uvicorn ASGI server on the specified host and port
- loads the FastAPI application from `src/mmore/run_index_api.py`

```{warning}
Keep this terminal window open. The backend runs in the foreground, and closing the terminal will shut it down.
```

#### 3. Concurrency and database

Uploads are processed by a background queue, one job per GPU (GPUs are auto-detected).
Two config fields tune this:

- `jobs_per_gpu` (default `1`): jobs processed per GPU at once. Total workers = GPUs * `jobs_per_gpu`.
- `max_queue_size` (default `null` = `num_gpu` * `jobs_per_gpu` * 10): pending-job cap, uploads beyond it get `503` http error.

GPUs are auto-detected, but you can restrict which (and how many) mmore uses with the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
# If you want to use only GPUs 0 and 2
CUDA_VISIBLE_DEVICES=0,2 python3 -m mmore index-api --config-file /path/to/config.yaml --host the_host --port the_port
```

```{note}
For `jobs_per_gpu > 1`, prefer a **Milvus Standalone** server
(`db.uri: http://localhost:19530`) over Milvus Lite. Milvus Standalone is better
suited to a production environment with concurrent load. Lite is fine for small or
local use. Keep `jobs_per_gpu: 1` with Lite.
```


## 📂 API Usage

### Upload endpoints

#### ▶️ `POST /v1/files`

**Upload a single file**

| Parameter | Type | Description |
| --- | --- | --- |
| `fileId` | `str` (form) | Unique identifier for the file |
| `file` | `UploadFile` (form) | File content to upload |
| `collectionName` | `str` (form, optional) | Collection to index into, created if missing |
- queues a background job, returns `202` with a `jobId`

**Response** (`202 Accepted`):

```json
{
  "jobId": "a1b2c3d4...",
  "fileId": "example123"
}
```

**Errors**: `409` file ID already exists or is being processed, `422` file without a name, `503` job queue full.


#### ▶️ `POST /v1/files/bulk`

**Upload multiple files with IDs**

| Parameter | Type | Description |
| --- | --- | --- |
| `listIds` | `List[str]` (form) | Comma-separated list of file IDs |
| `files` | `List[UploadFile]` (form) | Files to upload |
| `collectionName` | `str` (form, optional) | Collection to index all files into, created if missing |
- validates 1-to-1 correspondence between files and IDs
- queues **one independent job per file**, a bad file does not fail the batch

**Response** (`202 Accepted`):

```json
{
  "jobs": [
    {"fileId": "doc1", "jobId": "a1b2c3..."},
    {"fileId": "doc2", "error": "already exists"}
  ]
}
```

**Errors**: `400` number of IDs does not match number of files.


### 🔁 Update Endpoint

#### ✏️ `PUT /v1/files/{fileId}`

**Replace an existing file and re-index**

| Parameter | Type | Description |
| --- | --- | --- |
| `fileId` | `str` (path) | Existing file ID |
| `file` | `UploadFile` (form) | New file to replace with |
| `collectionName` | `str` (form, optional) | Collection the file is indexed in |
- queues a background job, returns `202` with a `jobId`
- old vectors are replaced only after the new content is processed (no data loss on failure)

**Response** (`202 Accepted`):

```json
{
  "jobId": "a1b2c3d4...",
  "fileId": "doc123"
}
```

**Errors**: `404` file or collection not found, `409` file already being processed, `422` file without a name, `503` job queue full.


### 🗑️ Delete endpoint

#### ❌ `DELETE /v1/files/{fileId}`

**Delete a file and remove its vector entry**

| Parameter | Type | Description |
| --- | --- | --- |
| `fileId` | `str` (path) | ID of the file to delete |
| `collectionName` | `str` (query, optional) | Collection the file is indexed in |
- deletes both local file and vector DB entry.

**Response**:

```json
{
  "status": "success",
  "message": "File successfully deleted",
  "fileId": "doc123"
}
```

**Errors**: `404` file or collection not found, `500` internal error while deleting the file (vector DB or filesystem).


### 📥 Download endpoint

#### 📄 `GET /v1/files/{fileId}`

**Download a file by its ID**

| Parameter | Type | Description |
| --- | --- | --- |
| `fileId` | `str` (path) | ID of the file to download |
| `collectionName` | `str` (query, optional) | Collection the file is indexed in |

Returns the file with binary content, named after the file's `filename` metadata.

**Errors**: `404` file or collection not found, `500` internal error while retrieving the file.

### 🏷️ Metadata endpoints

Metadata is stored on every chunk of a file, there are two kinds of keys:

- **file-level** metadata: keys present on all chunks of the file with the same value (e.g. `filename`). `GET` and `PATCH /v1/files/{fileId}/metadata` only deal with these, a file patch is applied to every chunk.
- **chunk-level** metadata: keys whose value differs from one chunk to another (e.g. `paragraph_positions`). They are not returned by the file `GET` and cannot be patched file-wide, use `PATCH /v1/files/{fileId}/chunks/{chunkId}/metadata` to patch a single chunk.

Both patches are a merge: a new key is added, an existing key is updated and a key set to `null` is deleted.

#### 🔎 `GET /v1/files/{fileId}/metadata`

**Get the file-level metadata of a file**

| Parameter | Type | Description |
| --- | --- | --- |
| `fileId` | `str` (path) | ID of the file |
| `collectionName` | `str` (query, optional) | Collection the file is indexed in |

**Response**: the file-level metadata, `chunks` is the number of chunks of the file.

```json
{
  "fileId": "doc123",
  "collectionName": "my_docs",
  "chunks": 12,
  "metadata": {
    "filename": "report.pdf",
    "file_path": "./uploads/doc123"
  }
}
```

**Errors**: `404` file or collection not found.

#### 🩹 `PATCH /v1/files/{fileId}/metadata`

**Patch the file-level metadata of a file, without re-indexing**

| Parameter | Type | Description |
| --- | --- | --- |
| `fileId` | `str` (path) | ID of the file |
| `collectionName` | `str` (query, optional) | Collection the file is indexed in |
| body | JSON object | Merged into the metadata: new keys are added, existing keys are updated, `null` deletes a key, others are left untouched |

```bash
curl -X PATCH "http://localhost:8000/v1/files/doc123/metadata?collectionName=my_docs" \
  -H "Content-Type: application/json" \
  -d '{"filename": "renamed.pdf"}'
```

- the patch is applied to every chunk of the file, embeddings are kept as is
- the patch is a merge: a new key is added to every chunk, an existing key is updated, a key set to `null` is deleted from every chunk (deleting a missing key is a no-op)
- values must be JSON scalars (string, number, boolean)
- an existing key keeps its type (a string stays a string, a number a number)
- `id`, `document_id`, `text`, the embeddings, `file_path`, `processed_at` and `processor_type` are protected
- the chunks are rewritten in batches, a failure midway can leave the file partially patched: the patch is idempotent, send it again

**Response**: same shape as `GET /v1/files/{fileId}/metadata`, after the patch (`chunks` is the number of rewritten chunks).

```json
{
  "fileId": "doc123",
  "collectionName": "my_docs",
  "chunks": 12,
  "metadata": {
    "filename": "renamed.pdf",
    "file_path": "./uploads/doc123"
  }
}
```

**Errors**: `404` file or collection not found, `422` invalid patch (empty, protected key, wrong type, non-scalar value, or chunk-level key, including setting it to `null`), with the list of problems in `detail`.

#### 🧩 `PATCH /v1/files/{fileId}/chunks/{chunkId}/metadata`

**Patch the metadata of a single chunk, without re-indexing**

| Parameter | Type | Description |
| --- | --- | --- |
| `fileId` | `str` (path) | ID of the file |
| `chunkId` | `str` (path) | ID of the chunk within the file (as returned by `/v1/retrieve`) |
| `collectionName` | `str` (query, optional) | Collection the file is indexed in |
| body | JSON object | Merged into the chunk's metadata: new keys are added, existing keys are updated, `null` deletes a key, others are left untouched |

```bash
curl -X PATCH "http://localhost:8000/v1/files/doc123/chunks/3/metadata" \
  -H "Content-Type: application/json" \
  -d '{"reviewed": true}'
```

- only this chunk is rewritten, the other chunks of the file are left untouched, embeddings are kept as is
- same rules as the file patch: protected keys (they cannot be deleted either), JSON scalars or `null` to delete, an existing key keeps its type
- choosing the right key is the caller's responsibility: patching a file-level key (e.g. `filename`) on one chunk makes it differ across the chunks, so it is no longer returned by `GET /v1/files/{fileId}/metadata`

**Response**: every metadata of the chunk after the patch, chunk-level keys included.

```json
{
  "fileId": "doc123",
  "chunkId": "3",
  "collectionName": "my_docs",
  "metadata": {
    "filename": "report.pdf",
    "file_path": "./uploads/doc123",
    "paragraph_positions": [[120, 340]],
    "reviewed": true
  }
}
```

**Errors**: `404` chunk or collection not found, `422` invalid patch (empty, protected key, wrong type or non-scalar value), with the list of problems in `detail`.

### 🛰️ Job status endpoints

Uploads return a `jobId`. Track it with either endpoint:

#### 📊 `GET /v1/jobs/{jobId}`

One-shot status snapshot. Returns `404` if the job is unknown (e.g. expired).

```json
{
  "jobId": "a1b2c3...", "fileId": "doc1", "filename": "doc.pdf",
  "status": "done", "device": "cuda:0",
  "result": {"chunks": 12}, "error": null
}
```

`status` is one of `queued`, `processing`, `done`, `failed`.

#### 📡 `GET /v1/jobs/{jobId}/events`

Server-Sent Events stream. The server pushes each status change and closes when
the job is `done` or `failed`, so the client does not poll. Example with `curl`:

```bash
curl -N http://host:port/v1/jobs/a1b2c3.../events
```

```{note}
Job status is kept in memory and dropped a couple of hours after the job ends
(and on restart). The durable record of an indexed document is its presence in
the collection, not the `jobId`.
```

---

## 🔄 How it works

1. **Upload** → bytes are saved, a background job is queued, `202` + `jobId` returns at once
2. **Process** (on the job's assigned GPU) → the file is processed
   1. **Crawling**: files are parsed using `Crawler`
   2. **Dispatching**: files are dispatched to the proper processor using `Dispatcher`
   3. **Processing**: text, images, and metadata are extracted and returned as a `MultiModalSample`
3. **Indexing** → dense and sparse vectors are stored in Milvus
4. The permanent file copy is saved only after indexing succeeds, then the job is marked `done`

## 🧰 Developer notes

- vector database: **Milvus** via `pymilvus`.
- default embedding models:
    - dense: `sentence-transformers/all-MiniLM-L6-v2`
    - sparse: `splade`
- supported file types:

```text
.pdf, .docx, .pptx, .md, .txt, .xlsx, .xls, .csv, .mp4, .avi, .mov, .mkv, .mp3, .wav, .aac, .eml, .html, .htm
```
    

### 💡 Tips

- avoid duplicate `fileId` unless you are intentionally updating a file with `PUT`
- you can test endpoints via Swagger UI at `/docs`

## See also

- [Indexing](../getting_started/indexing.md)
- [Process](../getting_started/process.md)
- [RAG](../getting_started/rag.md)

