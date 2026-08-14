# 🗂️ Indexing 

## Overview

The `index` module handles the indexing and post-processing of data extracted from multimodal documents.

It builds an indexed vector store based on [Milvus](https://milvus.io/) and supports **hybrid retrieval**, combining both **dense** and **sparse** retrieval.

Different parts of the indexing pipeline can be customized through an inference indexing configuration file.


## 💡 TL;DR

The indexing workflow takes processed documents and turns them into searchable artifacts that can later be used for retrieval and RAG pipelines.

In practice, this means:

- loading processed document data
- generating dense and sparse representations
- storing them in a Milvus-based vector store
- preparing the collection for hybrid retrieval


## 💻 Minimal Example:
Here is a minimal example to index [processed documents](process.md).

### 1. Create a config file

Start from the example configuration file: [`examples/index/config.yaml`](https://github.com/EPFLiGHT/mmore/blob/main/examples/index/config.yaml).


Adjust it to match your setup and indexing needs.

### 2. Run the indexing command

Once the configuration file is ready, launch the indexing pipeline with:
```bash
python3 -m mmore index --config_file /path/to/config.yaml
```

## Notes

The indexing step assumes that your documents have already been processed.

If you have not done that yet, start with [Process](process.md).

## See also

- [Process](process.md)
- [RAG](rag.md)
