# 🚀 Quickstart

## Overview

This page helps you get mmore running quickly with a minimal workflow.

The goal is not to cover every configuration option, but to give you a first successful setup and a clear mental model of the main steps.

## What this quickstart covers

In a typical mmore workflow, you will:

1. install the project and its dependencies
2. prepare a small document collection
3. process the collection
4. build an index
5. run retrieval or a simple RAG workflow

## Before you start

Make sure you have already read [Installation](installation.md).

You should also confirm that:

- your environment is activated
- project dependencies are installed
- you are working on a small test collection first

## Minimal workflow

The exact commands depend on your repository entry points, but the overall workflow is the following.

### 1. Prepare a small collection

Start with a small and simple document set before moving to large-scale or distributed workloads.

For example, create a folder containing a few representative documents:

```text
sample_data/
├── doc1.pdf
├── doc2.pdf
├── doc3.html
└── doc4.md
```

### 2. Run document processing

Processing transforms raw documents into a form that mmore can index and retrieve from.

Depending on your setup, this step may include:

- parsing files
- extracting text and metadata
- chunking content
- preparing multimodal representations

See [Processing pipeline](process.md) for the detailed logic.

### 3. Build an index

Once documents are processed, create an index so they can be searched efficiently.

This step usually includes:

- selecting the indexing backend or strategy
- generating representations for chunks or documents
- storing the resulting index artifacts

See [Indexing](indexing.md) for the full indexing workflow.

### 4. Run retrieval

After indexing, you can test retrieval on a few example queries.

At this stage, you want to verify simple things:

- does the system return relevant documents?
- are the retrieved chunks meaningful?
- is the ranking roughly coherent?

### 5. Move to RAG if needed

If your workflow includes generation, retrieval results can then be passed into a RAG pipeline.

See [RAG](rag.md) for how retrieval and generation are combined.

## Example end-to-end flow

Conceptually, a first mmore run looks like this:

```text
Raw documents
    ↓
Processing
    ↓
Structured outputs / chunks / metadata
    ↓
Indexing
    ↓
Retrieval
    ↓
Optional RAG generation
```

## Recommended first checks

After your first run, verify the following:

- documents were correctly discovered and parsed
- processed outputs were actually generated
- the index was created where expected
- simple test queries return results
- retrieved content looks coherent and relevant

## Common mistakes

```{warning}
Do not start with a large or noisy collection.

When debugging a documentation-backed pipeline, a very small dataset is much easier to inspect and validate.
```

Typical first-run problems include:

- wrong environment or missing dependencies
- input paths that do not point to the expected collection
- outputs written to a different directory than expected
- indexing performed on incomplete processed data
- retrieval tested before the index is fully built

## Where to go next

After this page, the best next steps are:

1. [Architecture](architecture.md) to understand the big picture
2. [Processing pipeline](process.md) for ingestion and transformations
3. [Indexing](indexing.md) for indexing details
4. [RAG](rag.md) for retrieval-augmented generation
