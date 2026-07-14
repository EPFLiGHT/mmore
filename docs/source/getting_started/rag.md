# 🤖 RAG 

## Overview

The `rag` module enables the creation of a modular RAG inference pipeline for indexed multimodal documents.

It supports two main execution modes:

1. **API mode**: runs the pipeline as a server and exposes an API
2. **Batch mode**: runs inference from an input file of queries, for example a JSONL file

Different parts of the pipeline can be customized through a RAG inference configuration file.

## 💡 TL;DR

The RAG module lets you combine retrieval and generation over indexed multimodal documents.

In practice, it supports:

- a batch mode for file-based inference
- an API mode for serving the pipeline
- configurable retriever and LLM components
- optional WebRAG and CLI usage in batch mode

You can customize various parts of the pipeline by defining an inference RAG configuration file at
[`examples/rag/api/rag_api.yaml`](https://github.com/EPFLiGHT/mmore/blob/master/examples/rag/api/rag_api.yaml).


## 💻 Minimal Example:

Here is a minimal example to create a RAG pipeline hosted through [LangGraph](https://python.langchain.com/docs/langgraph/) servers.

### 1. Create a RAG inference config file

Create your RAG Inference config file based on the [batch example `examples/rag/config.yaml`](https://github.com/EPFLiGHT/mmore/blob/master/examples/rag/config.yaml) or the [API example `examples/rag/config_api.yaml`](https://github.com/EPFLiGHT/mmore/blob/master/examples/rag/config_api.yaml).

You can check the structure of the configuration file with the dataclass [RAGConfig]( https://github.com/EPFLiGHT/mmore/blob/master/src/mmore/rag/pipeline.py).

### 2. Start the RAG pipeline

Start your RAG pipeline using the `run_rag.py` script and your config file
```bash
python3 -m mmore rag --config-file /path/to/config.yaml
```

### 3. Query the server in API mode

In API mode, the RAG server exposes a health endpoint and a configurable RAG endpoint. By default, the RAG endpoint is `/rag`.

Check that the server is running:
```bash
curl --location --request GET http://localhost:8000/health
```

Send a RAG query:
```bash
curl --location --request POST http://localhost:8000/rag \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "What is Meditron?",
    "collection_name": "my_docs"
  }'
```
In batch mode, the pipeline is run directly with the input data specified in the configuration file, and the result is saved to the specified path.

See [`examples/rag`](https://github.com/EPFLiGHT/mmore/blob/master/examples/rag/) for other use cases.

## 🔎 Main modules

The RAG pipeline is built around two main modules:
1. The `Retriever`, which retrieves multimodal documents from the database. 
2. The `LLM`, which wraps different types of multimodal-able LLMs.

### Retriever

Here is an example on how to use the retriever module on its own. Note that it assumes that you already created a database using the [Indexing](indexing.md) workflow.

#### 1. Create a config

Start from the [example config file `examples/index/config.yaml`](https://github.com/EPFLiGHT/mmore/blob/master/examples/index/config.yaml).

#### 2. Retrieve from the vector store

```python
from mmore.rag.retriever import Retriever

# Create the Retriever
retriever = Retriever.from_config('/path/to/your/retriever_config.yaml')

# Retrieves the top 3 documents using an hybrid approach (e.g. dense + sparse embeddings)
retriever.retrieve(
    'What is Meditron?',
    k=3,
    collection_name="my_docs",
    search_type="hybrid"  # Options: "dense", "sparse", "hybrid"
)
```

### LLM

Here is an example on how to use the `LLM` module on its own. This also assumes that the indexing workflow has already been completed.

#### 1. Create a config file
```yaml
llm_name: gpt-4o-mini
max_new_tokens: 150
temperature: 0.7
```

#### 2. Query the LLM
```python
from mmore.rag.llm import LLM

# Create the LLM
llm = LLM.from_config('/path/to/your/llm_config.yaml')

# Create your messages
messages = [
(
    "system",
    "You are a helpful assistant that translates English to French. Translate the user sentence.",
),
(
    "human",
    "I love Meditron."
),
]

# Retrieves the top 3 documents using an hybrid approach (e.g. dense + sparse embeddings)
llm.invoke(messages)
```

## 🔧 Customization

Our RAG pipeline is built to take full advantage of [LangChain](https://python.langchain.com/docs/introduction/) abstractions, providing compatibility with all components offered.

#### Retriever

Our retriever is a LangChain [`BaseRetriever`](https://python.langchain.com/api_reference/core/retrievers/langchain_core.retrievers.BaseRetriever.html). If you want to create a custom retriever (e.g. GraphRetriever,...) you can simply make it inherit from this class and use it as described in our examples.

#### WebRAG 
Within the `rag` pipeline, web search is currently configured through the retriever settings in local / file-based workflows.

It uses the [`DuckDuckGo Search API`](https://python.langchain.com/docs/integrations/tools/ddg/) to search the web using the input query, then adds its results to the context. 

#### CLI for RAG 
A CLI is also available for interactive querying.

Start it with:

```bash
python3 -m mmore ragcli --config-file /path/to/config.yaml
```

You can customize the CLI by defining [a RAG configuration file](https://github.com/EPFLiGHT/mmore/blob/master/examples/rag/config.yaml) or by setting preferences from within the CLI.



#### LLM
The LLM wrappers are based on LangChain's [`BaseChatModel`](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html). 

If you want to create a custom retriever you can simply make it inherit from this class and use it as described in our examples. 

```{warning}
mmore supports [Hugging Face Hub](https://huggingface.co/models) models.

In some cases, a simpler solution is to push a model to the Hub and use it through the existing class rather than implementing a new wrapper.
```

## Notes

The standalone `websearch` module and the `rag` pipeline do not expose web search in exactly the same way.

In particular:

- the standalone `websearch` module supports API usage, with optional RAG integration
- within the `rag` pipeline, web search is currently configured through the retriever settings in local / file-based workflows
- file-based inference may be slow when using local models

## See also

- [Indexing](indexing.md)
- [Process](process.md)
- [Architecture](architecture.md)


