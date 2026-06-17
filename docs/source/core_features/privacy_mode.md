# Privacy mode

Privacy mode runs a multi-agent privacy pipeline between MMORE's retriever and
the answer model, so only sanitized, pre-cloud-verified context ever reaches the
LLM. It is query-time only: the vector DB stores the raw corpus unchanged, and
the pipeline operates on the retrieved top-k chunks of each request.

Enable it by pointing the `rag` command at a privacy config:

```bash
mmore rag --config-file examples/rag/config.yaml --privacy examples/rag/privacy.yaml
```

With the flag absent, the RAG chain, output schema, and saved results are exactly
as before. The flag carries the config path; there is no separate boolean and no
`enabled` key.

## Trust boundary

The pipeline runs the chain

```
analyzer -> detector -> sanitizer -> leakage_adversary -> (HITL gate) -> answer -> verifier -> report
```

over the retrieved chunks:

1. The **analyzer** picks the privacy domain and emits a per-request policy.
2. The **detector** runs the policy's PII engine over each raw chunk.
3. The **sanitizer** rewrites the chunks under the chosen strategy.
4. The **leakage adversary** attacks the sanitized context; on a detected leak
   it loops back to the analyzer to tighten the policy (bounded by
   `leakage_adversary.max_iterations`).
5. The **HITL gate** is the trust boundary. With `interactive: false` (batch /
   local) it auto-approves and the graph completes in one pass. With
   `interactive: true` it pauses for human approval before any context leaves
   for the answer model.
6. The **answer model** sees only the sanitized context, the query, and the
   domain prompt; it never reads the raw chunks.
7. The **advisory verifier** checks the answer (residual leakage, faithfulness)
   and raises type+count warnings. It is advisory only and does not loop back.

## Configuration

`privacy.yaml` is loaded directly as `PrivacyConfig`, so its fields sit at the
top level (no `privacy:` wrapper). See `examples/rag/privacy.yaml` for a full
example. Key fields:

- `domain` — `global` | `healthcare` | `humanitarian`; omit to let the analyzer
  infer it.
- `interactive` — the HITL gate; set `false` for batch / local runs.
- `detection.engine` — one engine: `presidio` | `gliner` | `llm` | `openai_filter`.
- `sanitization.strategy` — `token_masking` | `entity_replacement` |
  `synthetic_rewrite` | `presidio`.
- `answer.llm` — any `LLMConfig` backend (API or self-hosted/vLLM).
- `verifier.checks` + `verifier.warn_threshold` — advisory checks over the answer.

Config errors surface eagerly at startup: a missing `answer.llm`, an unknown
`domain`, or an unregistered detection engine fail before any query runs.

## Report schema (for operators)

Each request appends one PII-free `ReportRecord`, surfaced on the RAG output as
`privacy_report` (and `privacy_warnings`, the advisory type+count summary). It
records types and counts only, never raw content or PII:

| Field | Meaning |
| --- | --- |
| `request_id`, `timestamp` | request identity |
| `domain` | resolved privacy domain |
| `detection_engine` | engine used |
| `detection` | span count + per-entity-type counts |
| `sanitization_strategy` | strategy applied |
| `escalation_iterations`, `gate_iterations` | escalation-loop and gate counters |
| `gate_outcome` | `approved` / `re-looped` / `aborted` / `rejected` |
| `answer_backend`, `answer_model` | which model answered |
| `advisory_warnings` | verifier warnings as kind + count |
| `hitl` | gate event (fired, decision, hashed feedback) |
| `outcome` | `returned` / `returned-with-warnings` / `aborted-unsafe` |

## See also

- [RAG](../getting_started/rag.md)
- [LLM as a judge](llm_as_a_judge.md)
