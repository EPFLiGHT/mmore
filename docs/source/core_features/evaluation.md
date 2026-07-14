# 📊 RAG Evaluation Pipeline

## Overview

The `RAG` module includes an evaluator that can assess the full RAG pipeline, from context retrieval to the final LLM output.

The evaluation workflow consists of four main steps:

1. prepare a benchmark evaluation dataset in the required format
2. choose the metrics to evaluate
3. configure the evaluator, indexer, and RAG pipeline
4. run the evaluation for the selected retriever and LLM setup

mmore relies on [RAGAS](https://docs.ragas.io/) for evaluation. RAGAS is a library designed for evaluating LLM applications.

## 💡 TL;DR

The evaluator lets you measure both retrieval quality and answer quality in a single workflow.

In practice, this means:

- loading an evaluation dataset
- configuring metrics and models
- building the evaluation index
- running the RAG pipeline against benchmark queries
- computing evaluation scores with RAGAS

See the [available RAGAS metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/).


## 💻 Minimal Example

Here's a step-by-step guide to set up the evaluation pipeline:

### 1. Create the evaluator config file

This file defines the evaluation settings for your pipeline.

```yaml
hf_dataset_name: "Mallard74/eval_medical_benchmark"  # Hugging Face Eval dataset name (Example dataset)
split: "train"  # Dataset split
hf_feature_map: {'user_input': 'user_input', 'reference': 'reference', 'corpus': 'corpus', 'query_id': 'query_ids'}  # Column mapping
metrics:  # List of metrics to evaluate
  - LLMContextRecall
  - Faithfulness
  - FactualCorrectness
  - SemanticSimilarity
embeddings_name: "all-MiniLM-L6-v2"  # Evaluator Embedding model name
llm:  # Evaluator LLM config
  llm_name: "gpt-4o"
  max_new_tokens: 150
```

### 2. Create the indexer config file

This file configures the indexer during evaluation.

```yaml
dense_model_name: sentence-transformers/all-MiniLM-L6-v2
sparse_model_name: splade
db:
  uri: "./examples/rag/milvus_mock_eval_medical_benchmark.db"  # Dataset's Vectorstore URI
  name: "mock_eval_medical_benchmark"
chunker:
  chunking_strategy: sentence  # Your chunking strategy
```

### 3. Create the RAG pipeline config file

This file defines the RAG setup to evaluate.

```yaml
llm:
  llm_name: "gpt-4o-mini"  # RAG LLM model to evaluate
  max_new_tokens: 150
retriever:
  db:
    uri: "./examples/rag/milvus_mock_eval_medical_benchmark.db"  # Dataset's Vectorstore URI
  hybrid_search_weight: 0.5
  k: 3
```
    
### 4. Run the evaluation

Once the configuration files are in place, you can run the evaluation pipeline with the following Python script:

```python
from mmore.rag.evaluator import RAGEvaluator

# Instantiate RAGEvaluator
evaluator = RAGEvaluator.from_config(args.eval_config)

# Run the evaluation
result = evaluator(
    indexer_config=args.indexer_config,
    rag_config=args.rag_config
)
```

- See [`examples/rag/evaluation`](https://github.com/EPFLiGHT/mmore/blob/master/examples/rag/evaluation) for a simple example.
```{warning}
Create a separate database file for each evaluation dataset.

The pipeline creates partitions per dense model for convenience.
```

## 📦 Outputs

The evaluation run returns a result object containing the selected metric scores for the evaluated setup.

The exact structure depends on the evaluator configuration and selected metrics.

## See also

- [RAG](../getting_started/rag.md)
- [Indexing](../getting_started/indexing.md)
- [Process](../getting_started/process.md)
