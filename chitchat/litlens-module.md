# LitLens — Literature Analytics & Visualization Module — Design Spec

> **Codename: LitLens.** A literature-analytics lens — point it at a corpus of screened papers and it produces word clouds, screening dashboards, and per-source/per-topic plots.
>
> **Purpose of this document.** Design-spec / integration brief for an AI coding assistant (e.g. Claude) working in *another* codebase. Use it as the source of truth for porting the analytics layer out of CHITCHAT into a reusable module. The originating source files live in this repo under `analysis/`.

---

## 1. What LitLens does

Given a corpus of papers (`Paper[]`) and — optionally — a corresponding set of structured screening results, LitLens produces:

1. **Word clouds + frequency charts** per topical bucket, defined by boolean expressions.
2. **Screening dashboards** — priority distribution, publication timeline, technical-scope coverage, humanitarian-principle radar, methodology vs. ethics scatter.
3. **Source / venue / year analytics** for raw discovery corpora (before screening).
4. **Static PNGs + interactive HTML** side by side, plus a CSV/JSON summary.

LitLens is **post-pipeline**. It does not call any external API, does not download anything, and is fully reproducible from local JSON/JSONL inputs.

### When to use it
- You already have a `papers.json` (e.g. from the paper-discovery module).
- You optionally also have a screening output (per-paper structured scoring).
- You want figures to put in a report, a dashboard, or a slide deck.

### When NOT to use it
- For live, interactive exploration of a million-paper corpus → use a real BI tool, not LitLens.
- For statistical hypothesis testing → LitLens is descriptive; bring your own stats layer.

---

## 2. Three sub-modules (pick what you need)

LitLens is a **family of three independent analyzers**. The consumer codebase can install all or some.

```
                ┌─────────────────────────────────────────────────────┐
                │                       LitLens                       │
                ├──────────────────┬──────────────────┬───────────────┤
                │ litlens.cloud    │ litlens.scope    │ litlens.corpus│
                │ (word clouds)    │ (screening dash) │ (raw corpus)  │
                ├──────────────────┼──────────────────┼───────────────┤
   inputs       │ papers.json      │ screening.jsonl  │ papers.json   │
                │ + boolean        │                  │ + (opt)       │
                │   buckets        │                  │   boolean     │
                ├──────────────────┼──────────────────┼───────────────┤
   outputs      │ N×wordcloud.png  │ priority.png,    │ source/year/  │
                │ N×freq_chart.png │ timeline.html,   │ venue plots   │
                │ summary.csv      │ dashboard.html   │ network graphs│
                └──────────────────┴──────────────────┴───────────────┘
```

| Sub-module       | Source today (CHITCHAT)               | Inputs                                              | Outputs                                  |
|------------------|---------------------------------------|-----------------------------------------------------|------------------------------------------|
| `litlens.cloud`  | `analysis/word_cloud_analysis.py`     | `papers.json` + `boolean_combinations.json`         | N word-clouds + freq charts + summary    |
| `litlens.scope`  | `analysis/paper_analysis.py`          | `screening_results.jsonl`                           | Priority/timeline/heatmap/radar + dashboard |
| `litlens.corpus` | `analysis/web_scrape_analysis.py`     | `papers.json` (+ optional `unique_boolean_*.json`) | Source/year/venue plots                  |

---

## 3. Public API surface (recommended)

### 3.1 Python library

```python
from litlens import cloud, scope, corpus

# --- 3.1.1 Word clouds per boolean bucket ---
cloud.run(
    papers_path="output/papers.json",
    boolean_buckets_path="data/boolean_combinations.json",
    screening_path=None,                 # optional, used to filter to screened-in
    output_dir="reports/wordclouds/",
    top_n_words=100,
    min_word_length=3,
)

# --- 3.1.2 Screening-results dashboard ---
scope.run(
    screening_jsonl="output/screening_results_2026-05-22.jsonl",
    output_dir="reports/screening/",
    interactive=True,                    # also emit Plotly HTML
)

# --- 3.1.3 Raw-corpus analytics (pre-screening) ---
corpus.run(
    papers_path="output/papers.json",
    boolean_combinations_path="data/unique_boolean_combinations.json",
    output_dir="reports/corpus/",
)
```

Each `run()` returns a `RunResult` containing the output directory and a manifest of generated artifacts (paths + types). Programmatic consumers can use the manifest to embed plots in HTML reports.

### 3.2 CLI

```
litlens cloud   --papers papers.json --buckets boolean_combinations.json --out reports/wc/
litlens scope   --screening screening.jsonl --out reports/screen/
litlens corpus  --papers papers.json --out reports/corpus/
litlens all     --papers papers.json --screening screening.jsonl --out reports/
```

### 3.3 HTTP service (optional)

If wired up: each endpoint accepts an input artifact (uploaded JSON/JSONL) and returns a job-id; the job's result is a zip of PNGs+HTML+CSV.

| Method | Path                | Body                                | Returns      |
|--------|---------------------|-------------------------------------|--------------|
| POST   | `/v1/cloud`         | multipart (papers, buckets)         | `{job_id}`   |
| POST   | `/v1/scope`         | multipart (screening.jsonl)         | `{job_id}`   |
| POST   | `/v1/corpus`        | multipart (papers, boolean opts)    | `{job_id}`   |
| GET    | `/v1/jobs/{id}`     | —                                   | status + zip URL when done |

---

## 4. Data contracts

### 4.1 `Paper` (input — same as paper-discovery output)

```jsonc
{
  "title":          "string | null",
  "authors":        "string, comma-separated | null",
  "url":            "string | null",
  "abstract":       "string | null",
  "year":           "int | null",
  "extracted_text": "string | null"
}
```

### 4.2 Boolean bucket definitions (input for `cloud`)

```jsonc
[
  { "WORD": "Foundation model",
    "boolean_combination": "(\"foundation model\" OR \"LLM\" OR \"GPT\")" }
]
```

> **Note.** `cloud` uses **per-WORD** OR-buckets (one bucket per row, one PNG per bucket). `corpus` can optionally use **category-level** `unique_boolean_combinations.json` (`Combination_title` + AND of OR-groups) for cross-cutting analytics.

### 4.3 Screening result (input for `scope`)

JSONL, one paper per line. Shape (from CHITCHAT's `src/screen_papers.py`):

```jsonc
{
  "title":            "...",
  "final_priority":   "INCLUDE | EXCLUDE | MAYBE | UNKNOWN",
  "original_metadata":{ "year": 2024, "authors": "..." },
  "screening_results":{
    "priority_level":         "P0 | P1 | P2 | P3",
    "publication_quality":    { "venue_name": "...", "is_top_tier_venue": bool,
                                "publication_year": int, "citation_count": int,
                                "is_recent_promising": bool, "full_text_english": bool },
    "technical_scope":        { "addresses_llm_data_collection": bool,
                                "addresses_text_corpus_creation": bool,
                                "addresses_web_scraping_nlp": bool,
                                "addresses_multilingual_compilation": bool },
    "ethical_flags":          { "focuses_only_on_performance": bool,
                                "disregards_ethical_principles": bool,
                                "missing_ethical_approval": bool,
                                "violates_humanitarian_principles": bool },
    "humanitarian_principles":{ "humanity_score": int, "impartiality_score": int,
                                "independence_score": int, "neutrality_score": int },
    "methodology_contributions":{ "novel_methodology": bool, "systematic_evaluation": bool,
                                  "reproducible_implementation": bool },
    "ethical_contributions":  { "explicit_framework": bool, "empirical_bias_analysis": bool,
                                "harm_mitigation_strategies": bool,
                                "policy_recommendations": bool, "acknowledges_tensions": bool }
  }
}
```

> **Schema-agnostic mode (recommended for porting).** Hardcoding CHITCHAT's exact screening schema makes LitLens brittle in other contexts. Implement an **adapter** layer: a user-supplied function `flatten(record) -> dict` so any screening shape can feed `scope`. Ship the CHITCHAT flattener as the default but pluggable.

### 4.4 Outputs

All sub-modules write to `output_dir/` with this convention:

```
output_dir/
├── manifest.json                # list of artifacts produced
├── *.png                        # static plots
├── *.html                       # interactive Plotly plots
├── *.csv                        # tabular summaries
└── analysis_report.md           # human-readable summary (scope only)
```

`manifest.json` is the programmatic surface — never rely on listing the directory.

---

## 5. `litlens.cloud` — design details

Pipeline:

```
  papers.json  ────┐
                   │
  boolean_buckets ─┴──▶  BooleanSearchEngine.match
                                  │
                                  ▼
                        bucket → [paper, paper, ...]
                                  │
                                  ▼
                        TextPreprocessor.extract_meaningful_words
                                  │
                                  ▼
                        Counter → top-N word frequencies
                                  │
                                  ▼
                       WordCloud + barh frequency chart + CSV
```

### Components

| Class                | Responsibility                                                    |
|----------------------|-------------------------------------------------------------------|
| `BooleanSearchEngine`| Parse `(A OR B OR C)` expressions; case-insensitive substring match against `title + abstract + extracted_text`. |
| `PaperBucketing`     | Apply each bucket's expression across the corpus; a paper can land in multiple buckets. |
| `TextPreprocessor`   | Lowercase, strip URLs/emails/punctuation, tokenize, POS-tag, drop stopwords, drop generic-academic words, allowlist technical terms. |
| `WordCloudGenerator` | Render WordCloud + matplotlib bar chart per bucket. |
| `WordCloudAnalyzer`  | Orchestrator. |

### Stopwords / allowlist
The CHITCHAT preprocessor ships a **very large** custom stopword set (~600 entries) covering generic academic vocabulary (`paper`, `study`, `framework`, …), modal verbs, ordinals, time words. It also ships a **technical-term allowlist** that bypasses POS filtering so terms like `nlp`, `ai`, `transformer` survive.

> **Port these as data, not code.** Ship them as `stopwords_extra.txt` and `technical_terms.txt`. The consumer can override either without forking the module.

### NLTK setup
The module needs `punkt`, `punkt_tab`, `stopwords`, `averaged_perceptron_tagger` (and `_eng` on newer NLTK). On import, attempt download with `nltk.download(..., quiet=True)` and **fall back to regex tokenization** if NLTK fails. Never let a missing NLTK resource crash the module.

### Match semantics — important
Bucket matching is currently **OR-only substring** — the expression `(A OR B)` matches if any term appears in concatenated `title+abstract+extracted_text`. The code does **not** implement `AND` / `NOT` for bucketing even though the boolean strings may contain them. If the consumer needs full boolean evaluation, plug in a real expression parser (e.g. `pyparsing`); otherwise document the limitation prominently.

### Knobs

| Knob              | Default | Notes |
|-------------------|---------|-------|
| `top_n_words`     | 100     | per bucket |
| `min_word_length` | 3       | drop 1-2 char tokens |
| `top_n_bars`      | 20      | how many words in the bar chart |
| `wordcloud_size`  | 1200×800 | render resolution |
| `colormap`        | viridis  | matplotlib name |
| `colorblind_safe` | False    | when true, force a CB-safe palette |

---

## 6. `litlens.scope` — design details

Operates on a **flattened DataFrame** built from the screening JSONL.

### Flattening
Walk each `screening_results` sub-object and project nested booleans/ints into a flat row. The CHITCHAT implementation is in `paper_analysis.flatten_screening_data`. Derived columns:

- `method_score`         = sum of methodology booleans
- `ethical_score`        = sum of ethical contributions booleans
- `total_humanitarian_score` = sum of the four humanitarian-principle integer scores

### Plots produced

| Function                          | Static PNG                       | Interactive HTML                       |
|-----------------------------------|----------------------------------|----------------------------------------|
| `plot_priority_distribution`      | `priority_distribution.png`      | `priority_distribution_interactive.html` |
| `plot_publication_analysis`       | `publication_analysis.png`       | `publication_timeline_interactive.html` |
| `plot_technical_scope_analysis`   | `technical_scope_analysis.png`   | `technical_scope_sunburst.html` |
| `plot_humanitarian_scores`        | `humanitarian_analysis.png`      | `humanitarian_radar.html` |
| `plot_methodology_contributions`  | `contributions_analysis.png`     | `contributions_bubble.html` |
| `plot_comprehensive_dashboard`    | —                                 | `comprehensive_dashboard.html` (3×2 multi-panel) |
| `generate_summary_stats`          | `summary_statistics.json`, `analysis_report.md` | — |

### Defensive plotting
Every plot must gracefully handle:
- **Empty filtered subsets** (e.g. no papers with `publication_year > 0`). The CHITCHAT code annotates the axes with "No valid X data" rather than throwing.
- **Mono-class data** (e.g. all `is_top_tier_venue=False`). Don't show a "True" slice with zero.
- **NaN in numeric scores** — use `pd.notna` checks before formatting in the markdown report.

Keep these guards in any port. They are the difference between a polished module and a fragile one.

### Schema-agnostic mode
If the consumer's screening schema differs, accept `flatten: Callable[[dict], dict]` as a parameter and dispatch to it instead of the built-in flattener. The plot functions only need the **derived flat columns** they read (document each plot's required columns).

---

## 7. `litlens.corpus` — design details

Operates on the **raw discovery output** (no screening required). Useful sanity-check / EDA before running a screening pass.

### Plots produced (from `web_scrape_analysis.py`)
- Papers per source (`url` → domain → source name).
- Papers per year (line + histogram), with optional smoothing.
- Top-N venues / domains.
- Author counts and co-authorship combinations (`itertools.combinations`).
- Boolean-category coverage heatmap (how many papers match each `Combination_title`).
- Interactive sunburst: `source → year → category`.

### Source derivation
`_extract_source(url)` does dynamic domain → friendly-name mapping. Externalize the mapping as a JSON file so consumers can rename / re-bucket sources without code changes.

### Year coercion
Already messy across the four sources (int / string / date-prefix). Reuse a shared `coerce_year(value) -> int | None` helper across `corpus` and `scope`.

---

## 8. Dependencies (minimum viable port)

```
matplotlib   >= 3.8
seaborn      >= 0.13
plotly       >= 5.20
pandas       >= 2.3
numpy        >= 2.0
wordcloud    >= 1.9       # cloud only
nltk         >= 3.9       # cloud only
jsonlines    >= 4.0       # scope only
```

**Drop when porting:** all CHITCHAT screening / discovery deps (openai, PyPDF2, scholarly, selenium, fitz). LitLens is pure analytics.

---

## 9. Suggested file layout in the consumer repo

```
litlens/
├── __init__.py              # re-exports cloud.run, scope.run, corpus.run, RunResult
├── config.py                # dataclass: paths, knobs, color palette
├── io.py                    # papers/screening loaders + manifest writer
├── flatten.py               # default screening flattener (CHITCHAT shape) + adapter protocol
├── text/
│   ├── __init__.py
│   ├── boolean_match.py     # BooleanSearchEngine
│   ├── preprocess.py        # TextPreprocessor + NLTK bootstrap
│   ├── stopwords_extra.txt  # data file
│   └── technical_terms.txt  # data file
├── cloud/
│   ├── __init__.py          # run()
│   ├── bucketing.py
│   └── render.py            # WordCloudGenerator
├── scope/
│   ├── __init__.py          # run()
│   ├── plots.py             # the 6 plot_* functions
│   └── report.py            # generate_summary_stats / markdown report
├── corpus/
│   ├── __init__.py          # run()
│   ├── analyzer.py
│   └── source_map.json      # domain → source friendly name
├── cli.py                   # `litlens` entrypoint with sub-commands
└── service/                 # optional FastAPI wrapper
```

---

## 10. Output conventions (consistency across sub-modules)

- **Static images**: PNG at 300 dpi, `bbox_inches='tight'`, white facecolor.
- **Interactive plots**: Plotly HTML, self-contained (`include_plotlyjs='cdn'`).
- **Tables**: CSV + JSON pair where it makes sense.
- **Markdown report**: only `scope` writes one — `analysis_report.md` with priority distribution, technical-scope coverage, humanitarian averages.
- **Manifest**: every `run()` writes `manifest.json`:
  ```jsonc
  {
    "module": "scope",
    "generated_at": "2026-05-22T14:03:11Z",
    "inputs": { "screening_jsonl": "..." },
    "artifacts": [
      {"path": "priority_distribution.png",     "kind": "image/png"},
      {"path": "priority_distribution_interactive.html", "kind": "text/html"},
      ...
    ]
  }
  ```

---

## 11. Configuration knobs

Centralize in a `LitLensConfig` dataclass. Each `run()` accepts an instance or overrides as kwargs.

| Knob                    | Default | Used by         | Notes |
|-------------------------|---------|-----------------|-------|
| `dpi`                   | 300     | all             | static images |
| `figsize_default`       | (12, 8) | all             | |
| `colormap`              | viridis | cloud           | wordcloud + bars |
| `palette`               | husl    | scope, corpus   | seaborn |
| `colorblind_safe`       | False   | all             | force CB-safe palette |
| `top_n_words`           | 100     | cloud           | per bucket |
| `min_word_length`       | 3       | cloud           | |
| `min_papers_per_bucket` | 1       | cloud           | skip tiny buckets |
| `interactive`           | True    | scope, corpus   | emit Plotly HTML |
| `write_manifest`        | True    | all             | |
| `stopwords_extra_path`  | bundled | cloud           | override custom list |
| `technical_terms_path`  | bundled | cloud           | override allowlist |

---

## 12. Errors, edge cases, things that bite

1. **NLTK resource missing.** Wrap `pos_tag` / `word_tokenize` in try/except; fall back to regex `\b[a-zA-Z]+\b`. Never crash on `LookupError`.
2. **Empty bucket.** Don't emit empty PNGs; log a warning and skip. Reflect in manifest as `"skipped": true`.
3. **Single-class boolean column.** `seaborn.barplot` warns if a class is absent; build `labels`/`values` only from present classes (see `plot_publication_analysis` for the pattern).
4. **No valid years.** `df[df['publication_year'] > 0]` may be empty; annotate the axis instead of plotting nothing.
5. **JSON vs JSONL.** `corpus` reads `json.load` (single array); `scope` reads JSON-Lines. Don't mix.
6. **Plotly font size on Windows.** Default Plotly font may render poorly in some Windows browsers — set an explicit font family in the layout.
7. **Stopwords list drift.** The CHITCHAT custom stopword set is opinionated for AI/humanitarian literature. Externalize it (§9) so other domains can swap.
8. **Boolean expressions with AND/NOT.** `BooleanSearchEngine` silently treats them as OR-only. Document this loudly or upgrade the parser.
9. **`is_top_tier_venue` etc. are bools, not nullable.** If your screening can produce `null`, coerce to `False` (or to a third "unknown" category) before plotting.
10. **Plot subdirectories.** Don't create per-bucket subdirs — flatten with `_make_safe_filename` so manifests stay simple.

---

## 13. Extension points

- **New plot.** Drop a `plot_xyz(df, output_dir)` into `scope/plots.py` and register it in `scope.run`'s plot list.
- **Custom screening schema.** Pass `flatten=my_flattener` into `scope.run`. The flattener must produce the columns referenced by whichever plots the consumer enables.
- **New analyzer.** Add a `litlens.<name>/` sibling with the same `run()` contract; plug into `cli.py`.
- **Theming.** Replace the default palette/colormap via `LitLensConfig`. Provide a `colorblind_safe=True` shortcut.
- **Embedding into a Jupyter notebook.** Each `plot_*` function should return the `Figure` *and* save to disk. Today the CHITCHAT versions only save — refactor on port.
- **Per-paper drill-through.** Add `--include-paper-ids` so interactive plots' hover-data include the source paper IDs for downstream linking.

---

## 14. Integration checklist (for the assistant in the consumer repo)

- [ ] Decide which of the three sub-modules the host project actually needs; skip the others.
- [ ] Copy `analysis/word_cloud_analysis.py` → `litlens/cloud/` (split into `bucketing.py`, `render.py`, plus shared `text/`).
- [ ] Copy `analysis/paper_analysis.py` → `litlens/scope/`. Split the six `plot_*` functions into `plots.py` and the report builder into `report.py`.
- [ ] Copy `analysis/web_scrape_analysis.py` → `litlens/corpus/`.
- [ ] Externalize the **stopwords list, technical-term allowlist, and source domain map** as data files in the package.
- [ ] Externalize the **CHITCHAT-specific screening schema** by introducing the `flatten` adapter (§4.3).
- [ ] Replace all hardcoded `output_dir = "analysis/plots/..."` strings with `config.output_dir`.
- [ ] Refactor each `plot_*` to return its `Figure` and accept a `config` arg, in addition to writing to disk.
- [ ] Add `manifest.json` emission at the end of each `run()`.
- [ ] Add `litlens` CLI with `cloud` / `scope` / `corpus` / `all` sub-commands.
- [ ] Write one smoke test per sub-module with a tiny synthetic dataset.
- [ ] If running headless, set the matplotlib backend to `Agg` at module init.

---

## 15. Reference: source files in CHITCHAT

| Source file (CHITCHAT)                       | Maps to                                    |
|----------------------------------------------|--------------------------------------------|
| `analysis/word_cloud_analysis.py`            | `litlens/cloud/*`, `litlens/text/*`        |
| `analysis/paper_analysis.py`                 | `litlens/scope/plots.py`, `scope/report.py`|
| `analysis/web_scrape_analysis.py`            | `litlens/corpus/analyzer.py`               |

---

## 16. Why "LitLens"?

Same idea as a camera lens: point it at a literature corpus and it brings the salient structure into focus — what's there, what's missing, what got prioritized, what got dropped. Three lenses (`cloud`, `scope`, `corpus`) for three focal lengths.

End of spec.
