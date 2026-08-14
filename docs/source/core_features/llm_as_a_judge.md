# LLM as a judge

Add a `judge:` block to your RAG config to check retrieval quality before generation. When chunks are not good enough, the judge can trigger a corrective action such as re-search, sub-questions, or web context and then merge everything into one deduplicated list.

## How it works

1. Retrieve chunks from the index (Milvus + optional BGE rerank).
2. Evaluate them in a loop (at most `max_corrective_steps` corrective actions, default `1`):
  - **`PROCEED` without calling the judge LLM** when index metrics meet `metric_thresholds` — retrieval is already considered good enough.
  - Otherwise, call `judge.llm` and run the chosen corrective action.
  - Repeat on the merged chunks until the judge says `PROCEED` or the step budget is exhausted.
3. Generate the answer from the final context.

Disallowed decisions are coerced to a fallback action (`RE_RETRIEVE`, `ADD_QUESTIONS`, or `PROCEED`). Invalid JSON defaults to `PROCEED`.

## Decisions


| Decision        | What it does                                                    |
| --------------- | --------------------------------------------------------------- |
| `PROCEED`       | Chunks are good enough; continue to the answer LLM              |
| `RE_RETRIEVE`   | Search the index again (reformulated query and/or more results) |
| `ADD_QUESTIONS` | Up to 3 extra searches from sub-questions, then merge           |
| `ADD_CONTEXT`   | DuckDuckGo web snippets, then merge                             |


## Configuration

`examples/rag/config_judge.yaml` is a standalone config — it does not load on top of `config.yaml`.

```bash
python3 -m mmore rag --config-file examples/rag/config_judge.yaml
```

Or copy the `judge:` block into your own config.

Key settings under `rag.judge`:

- `metric_thresholds` — index minimums (`min_mean_similarity`, `min_max_rerank_score`, `min_num_docs`, …)
- `max_corrective_steps` — how many corrective actions after the first retrieval
- `allow_re_retrieve` / `allow_add_questions` / `allow_add_context` — which corrective actions the judge may choose (see below)
- `system_prompt` / `user_prompt` — judge prompts; user prompt supports `{query}`, `{metrics}`, `{chunks}`, `{allowed_actions}`, and correction-step placeholders

### Using one corrective action

In your RAG config, under `**rag.judge**`, set `allow_re_retrieve`, `allow_add_questions`, and `allow_add_context` so **only one** corrective action is `true` (the others `false`). `PROCEED` is always available in `{allowed_actions}`.

When `**metric_thresholds` are met**, the pipeline `**PROCEEDS` immediately** without calling the judge LLM: index retrieval is already of high quality (similarity, rerank scores, enough documents).

When **thresholds fail**, the judge LLM is invoked. With a single corrective action enabled, it **systematically chooses that action** and fills the matching payload (`extra_questions`, `web_query`, or `retrieve_params`). Use a query suited to that action (multi-part question → `ADD_QUESTIONS`; missing corpus fact → `ADD_CONTEXT`; weak or mis-phrased retrieval → `RE_RETRIEVE`). Adjust `system_prompt` / `user_prompt` under `rag.judge` if needed.


| Goal                            | `rag.judge` `allow_`* settings                                                      |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| Sub-questions (`ADD_QUESTIONS`) | `allow_add_questions: true`, `allow_add_context: false`, `allow_re_retrieve: false` |
| Web context (`ADD_CONTEXT`)     | `allow_add_context: true`, `allow_add_questions: false`, `allow_re_retrieve: false` |
| Re-retrieval (`RE_RETRIEVE`)    | `allow_re_retrieve: true`, `allow_add_questions: false`, `allow_add_context: false` |


Examples: `examples/rag/demo/config_add_questions.yaml`, `config_judge_add_questions.yaml`.

For `ADD_CONTEXT`, install web search support:

```bash
pip install "mmore[rag,websearch]"
```

## See also

- [RAG](../getting_started/rag.md)
- [Websearch](websearch.md)

