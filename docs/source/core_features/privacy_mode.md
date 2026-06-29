# Privacy mode

Privacy mode adds a multi-agent privacy pipeline between MMORE's retriever and the answer model, so only cleaned and checked context ever reaches the LLM. It runs at query time only: the vector DB keeps the raw corpus unchanged, and the pipeline works on the top-k chunks retrieved for each request.

Turn it on by pointing the `rag` command at a privacy config:

```bash
mmore rag --config-file examples/rag/config.yaml --privacy examples/rag/privacy.yaml
```

Without the flag, the RAG chain, output schema, and saved results stay exactly as before. The flag itself carries the config path; there is no separate boolean and no `enabled` key.

## Trust boundary

The pipeline runs this chain over the retrieved chunks:

```
analyzer -> detector -> sanitizer -> leakage_adversary -> (HITL gate) -> answer -> verifier -> report
```

1. The **analyzer** picks the privacy domain and creates a policy for that request.
2. The **detector** runs the policy's PII engine over each raw chunk.
3. The **sanitizer** rewrites the chunks using the chosen strategy.
4. The **leakage adversary** attacks the sanitized context. If it finds a leak, it loops back to the analyzer to tighten the policy (limited by `leakage_adversary.max_iterations`).
5. The **HITL gate** is the trust boundary. With `interactive: false` (batch or local) it approves automatically and the graph finishes in one pass. With `interactive: true` it pauses for human approval before any context leaves for the answer model.
6. The **answer model** sees only the sanitized context, the query, and the domain prompt. It never reads the raw chunks.
7. The **verifier** checks the answer for leftover PII and faithfulness, and raises type and count warnings. It is advisory only: it warns but does not loop back.

## Configuration

`privacy.yaml` is loaded directly as `PrivacyConfig`, so its fields sit at the top level (no `privacy:` wrapper). See `examples/rag/privacy.yaml` for a full example. Main fields:

- `domain`: `global`, `healthcare`, or `humanitarian`. Leave it out to let the analyzer guess it.
- `interactive`: the HITL gate. Set `false` for batch or local runs.
- `detection.engine`: one engine, either `presidio`, `gliner`, `llm`, or `openai_filter`.
- `sanitization.strategy`: `token_masking`, `entity_replacement`, `synthetic_rewrite`, or `presidio`.
- `answer.llm`: any `LLMConfig` backend (API or self-hosted/vLLM).
- `verifier.checks` and `verifier.warn_threshold`: the advisory checks run over the answer.

Config errors are reported at startup: a missing `answer.llm`, an unknown `domain`, or an unregistered detection engine fail before any query runs.

## Report schema (for operators)

Each request adds one PII-free `ReportRecord`, shown on the RAG output as `privacy_report` (plus `privacy_warnings`, the advisory type and count summary). It records types and counts only, never raw content or PII:

| Field | Meaning |
| --- | --- |
| `request_id`, `timestamp` | request identity |
| `domain` | resolved privacy domain |
| `detection_engine` | engine used |
| `detection` | span count and per-entity-type counts |
| `sanitization_strategy` | strategy applied |
| `escalation_iterations`, `gate_iterations` | escalation-loop and gate counters |
| `gate_outcome` | `approved`, `re-looped`, `aborted`, or `rejected` |
| `answer_backend`, `answer_model` | which model answered |
| `advisory_warnings` | verifier warnings as kind and count |
| `hitl` | gate event (fired, decision, hashed feedback) |
| `outcome` | `returned`, `returned-with-warnings`, or `aborted-unsafe` |

## See also

- [RAG](../getting_started/rag.md)
- [LLM as a judge](llm_as_a_judge.md)
