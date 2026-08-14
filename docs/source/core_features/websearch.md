# 🌐 WebSearch Integration

## Overview
The WebSearch integration uses the `ddgs` (DuckDuckGo Search) library with optional Tavily support. The implementation combines:

- **DuckDuckGo Search:** default provider — free, no API key needed. Install with `pip install "mmore[rag,websearch]"`.
- **Tavily Search:** ptional higher-quality provider. Requires `TAVILY_API_KEY` environment variable. Install with `pip install "mmore[rag,websearch]"`.
- **LLM Integration:** to summarize and integrate retrieved web snippets with RAG results to provide a final answer

This page describes the standalone `websearch` module.  
It should be distinguished from web search used inside the `rag` pipeline, which is configured separately through the RAG retriever settings.

## Installation & Search Providers

Install web search dependencies:
```bash
pip install "mmore[rag,websearch]"
```

| Provider | Default | API Key Required | Notes |
|---|---|---|---|
| DuckDuckGo | ✅ Yes | ❌ No | Free, no setup needed |
| Tavily | ❌ No | ✅ Yes (`TAVILY_API_KEY`) | Higher quality results |

To use Tavily, set the environment variable:
```bash
export TAVILY_API_KEY=your_key_here
```

Configure the provider in your config file:
```yaml
websearch:
    search_provider: "tavily"   # or "duckduckgo" (default)
    max_retries: 3              # retries on rate limit
```

## 🔧 Customization

Based on the implementation of the `RAG` module, the `Websearch` module enables the creation of a modular Websearch inference pipeline for indexed multimodal documents.  
It supports two inference modes:
1. **API**: runs a server hosting the pipeline
2. **Local**: runs the inference locally
 
You can customize various parts of the pipeline by defining a WebSearch inference configuration file such as [`examples/websearch/config_api.yaml`](https://github.com/EPFLiGHT/mmore/blob/main/examples/websearch/config_api.yaml).


Users can also adjust the pipeline through parameters defined in files such as [`examples/websearch/config.yaml`](https://github.com/EPFLiGHT/mmore/blob/main/examples/websearch/config.yaml).

Main parameters include:

- `use_rag`: enables or disables RAG retrieval.
- `use_summary`: enables summarization of retrieved web snippets.
- `n_loops`: defines the number of search iterations to refine results.
- `n_subqueries`: specifies the number of subqueries generated for each input query.
- `max_context_tokens`: maximum token budget for prompts (default: 2048).
- `fast_tokenizer`: if true, estimates tokens as ~4 chars/token instead of using the LLM tokenizer, faster but approximate (default: false).

## 🔁 Workflow

1. **Optional RAG pipeline**  
   If `use_rag` is enabled, the RAG pipeline is launched first and its output becomes the current knowledge.

2. **Input query processing**  
   The pipeline processes the user query and generates subqueries for web search in order to enrich the current knowledge.

3. **WebSearch execution**  
   Web searches are performed for each subquery using the configured provider.  

4. **Summarization**  
   Retrieved web snippets are summarized with an LLM if `use_summary` is enabled.

5. **Integration with RAG**  
   If `use_rag` is enabled, WebSearch results are combined with RAG outputs to update the current knowledge.

6. **Iteration**  
   The loop starts again from query processing using the updated knowledge, until the configured number of loops is reached.

## 💻 Minimal Example

Here is a example to create a Websearch pipeline hosted through [LangGraph](https://python.langchain.com/docs/langgraph/) servers.

### 1. Create a config file

Start from the [local example](https://github.com/EPFLiGHT/mmore/blob/main/examples/websearch/config.yaml) or the [API example](https://github.com/EPFLiGHT/mmore/blob/main/examples/websearch/config_api.yaml).

### 2. Start the WebSearch pipeline

```bash
python3 -m mmore websearch --config-file /path/to/config.yaml
```

### 3. Query the server in API mode
```bash
curl --location --request POST http://localhost:8000/websearch/ \
-H 'Content-Type: application/json' \
-d ' {
    "query": {
    "input": "When was Barack Obama born?",
    "collection_name": "my_docs"
    },
    "use_rag": true,
    "use_summary": true
    }
}'
```
In the API mode, `use_rag` and `use_summary` must be provided in the query.

The values of `n_loops` and `n_subqueries` are defined in the configuration file.

In local mode, the pipeline runs directly on the input data specified in the config file and the result is saved at the specified path.

In both modes, if `use_rag` is enabled, the path to a RAG configuration file must be provided.

## 📦 Results and Outputs

### Output format
The pipeline provides outputs in the following structure:

- **Short Answer**: a concise response derived from combined RAG and WebSearch results
- **Detailed Answer**: a longer response with context from both sources
- **Sources**: a list of URLs and their respective titles in the format:
  ```json
    {"URL": ["Title 1", "Title 2", ...]}
  ```

### Example Output
```json
{
    "query": "What are the latest advancements in AI?",
    "rag_informations": "Pre-existing knowledge from RAG.",
    "rag_summary": "Summary of RAG knowledge.",
    "web_summary": "Summarized web search information.",
    "short_answer": "AI advancements include new models and real-time applications.",
    "detailed_answer": "Recent AI developments include breakthroughs in large language models and innovative real-time applications, supported by diverse resources from RAG and web search.",
    "sources": {
        "https://example1.com" : ["AI Breakthrough; Latest in AI"],
        "https://example2.com" : ["Advancements in AI; AI Trends"],
    }
}
```

## See also

- [RAG](../getting_started/rag.md)
- [Indexing](../getting_started/indexing.md)
- [Process](../getting_started/process.md)

