# 🪟 Running mmore on Windows

## Overview

mmore was developed and tested mainly on Linux. It runs on Windows too, but a few things behave differently. This page lists those differences and the fix for each one.

If you work on Linux or macOS, you can skip this page.

## 1. Install the prerequisites

Unlike most Linux distributions, Windows does not ship Python, Git, or FFmpeg.
Install them first with
[winget](https://learn.microsoft.com/windows/package-manager/winget/):

```powershell
winget install Python.Python.3.11
winget install Git.Git
winget install astral-sh.uv
winget install Gyan.FFmpeg
```

Then clone the repo and install mmore into a virtual environment:

```powershell
git clone https://github.com/EPFLiGHT/mmore.git
cd mmore
uv venv
.venv\Scripts\activate
uv pip install -e ".[all,cu126]"
```

Use `cu126` for an NVIDIA GPU, or `cpu` otherwise. See the
[README](https://github.com/EPFLiGHT/mmore#step-1--install-mmore) for the full
list of extras.

## 2. `milvus-lite` is not available on Windows

Every example config whose `db.uri` is `./proc_demo.db` relies on `milvus-lite`
(`examples/index/config.yaml`, `examples/retriever_api/config.yaml`,
`examples/rag/config.yaml`, `examples/rag/config_api.yaml`). There is no Windows
build of `milvus-lite`, so any of them fails with:

```
ModuleNotFoundError: No module named 'milvus_lite'
```

### Fix: run Milvus in Docker

This repo ships no Compose file, so download the official Milvus standalone one
matching your installed `pymilvus` version (see the
[Milvus install docs](https://milvus.io/docs/install_standalone-docker-compose.md)):

```powershell
# Download the Milvus docker compose file from GitHub
Invoke-WebRequest `
  -Uri "https://github.com/milvus-io/milvus/releases/download/v2.6.6/milvus-standalone-docker-compose.yml" `
  -OutFile "milvus-docker-compose.yml"
# Start Milvus containers
docker compose -f milvus-docker-compose.yml up -d
```

Wait about a minute, then check `docker ps` shows the three containers
(`etcd`, `minio`, `milvus-standalone`) as `(healthy)`.

### Create the database

mmore does not create the database automatically when connecting to a remote Milvus. Run this once:

```powershell
python -c "from pymilvus import connections, db; connections.connect(uri='http://127.0.0.1:19530'); db.create_database('my_db')"
```

### Point the configs at the Docker instance

The `db` block lives at a different level depending on the config. Change
`uri: ./proc_demo.db` to `uri: http://127.0.0.1:19530` in each one you use.

`examples/retriever_api/config.yaml` (and `examples/rag/config*.yaml`) — `db`
is at the root:

```yaml
db:
  uri: http://127.0.0.1:19530
  name: my_db
```

`examples/index/config.yaml` — `db` is nested under `indexer`:

```yaml
indexer:
  db:
    uri: http://127.0.0.1:19530
    name: my_db
```

### Check that the setup works

Once Milvus is running, confirm the connection:

```powershell
python -c "from pymilvus import MilvusClient; c = MilvusClient(uri='http://127.0.0.1:19530', db_name='my_db'); print(c.list_collections())"
```

This returns a list of collections (empty before you index anything).

## 3. Surya OCR can crash the process on large PDFs

When processing large PDFs, the surya-based OCR may crash with:

```
Process finished with exit code 0xC0000005
```

This is a hard crash inside a native dependency. On Windows, use the fast processors instead, which rely on PyMuPDF rather than surya.

In your `process` config, `use_fast_processors` goes under `dispatcher_config`:

```yaml
dispatcher_config:
  use_fast_processors: true
```

You lose some accuracy on heavily scanned PDFs, but the pipeline no longer crashes.

## See also

- [Installation](installation.md)
- [Quickstart](quickstart.md)
