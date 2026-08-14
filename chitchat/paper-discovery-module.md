# Paper Discovery Module — Design Spec

> **Purpose of this document.** This is a **design-spec / integration brief** for an AI coding assistant (e.g. Claude) working in *another* codebase. It tells the assistant exactly what to lift out of CHITCHAT, how the pieces fit together, what the public surface should look like, and where the rough edges are. Implement the module in the consumer repo by following this doc end-to-end; the source files referenced are in this repo under `src/`.

---

## 1. What this module does

A standalone, reusable component that turns a **flat list of domain keywords + synonyms** into a **deduplicated list of research papers** (with metadata + extracted full text) by:

1. Expanding each keyword into an `OR`-group of quoted synonyms ("Boolean expansion").
2. Composing keyword groups into N category-level `AND`-queries.
3. Federating each query across multiple academic repositories (OpenAlex, Europe PMC, arXiv, Google Scholar).
4. Downloading any reachable PDFs and extracting their text.
5. Returning a single normalized JSON list.

It has **no opinion about what comes next** (screening, ranking, embedding, citation graph, …). The output is a clean `Paper[]` schema that any downstream consumer can use.

### When to use it
- You have a domain ontology (CSV/JSON of `WORD`, `SYNONYMS`) and want literature on it.
- You want to abstract away the four different repository APIs and their quirks.
- You want PDF text where available, but graceful degradation (abstract-only) where not.

### When NOT to use it
- Single-keyword searches → call the repository API directly.
- Real-time / sub-second search → this module batches and rate-limits, expect minutes per category.
- Citation-graph traversal → out of scope.

---

## 2. Pipeline (data flow)

```
            synonyms.csv / structure.json
                       │
                       ▼
   ┌───────────────────────────────────────┐
   │  STAGE 1 — Boolean Query Builder      │   (deterministic, offline)
   │                                       │
   │   per-WORD  OR-group of "synonym"s    │  → boolean_combinations.json
   │   per-category  AND of OR-groups      │  → unique_boolean_combinations.json
   └───────────────────────────────────────┘
                       │
                       ▼
   ┌───────────────────────────────────────┐
   │  STAGE 2 — Multi-Source Paper Fetch   │   (network, rate-limited)
   │                                       │
   │   dispatcher → OpenAlex                │
   │              → Europe PMC              │
   │              → arXiv                   │
   │              → Google Scholar          │
   │   ↓                                    │
   │   PDF download → text extraction       │
   └───────────────────────────────────────┘
                       │
                       ▼
                  papers.json
```

The two stages are **independently usable**. Stage 1 produces a portable artifact (the query JSON) that can be inspected, version-controlled, or fed into any other search tool.

---

## 3. Public API surface (recommended)

The module ships **three calling layers** — pick whichever fits the host codebase:

### 3.1 Python library (primary)

```python
from paper_discovery import (
    build_boolean_queries,        # Stage 1
    discover_papers,              # Stage 1 + Stage 2 end-to-end
    search_source,                # Stage 2 only, single source
    search_all_sources,           # Stage 2 only, fan-out
)

# Stage 1 only — pure, offline, deterministic.
queries: list[CategoryQuery] = build_boolean_queries(
    synonyms=[{"WORD": "Foundation model",
               "SYNONYMS AND NEAR SYNONYMS": "LLM, large language model, GPT"}],
    categories={
        "Broad Foundational Search": ["Foundation model", "Machine Learning"],
        "Humanitarian Search":       ["Humanitarian & Crisis Response", ...],
    },
)
# → [{"Combination_title": "...", "boolean_combination": '("LLM" OR ...) AND (...)'}]

# Stage 2 only — given a query, fetch papers from one source.
papers = search_source(query='("LLM" OR "GPT") AND "humanitarian"',
                       source="openalex",
                       max_pages=5)

# End-to-end.
papers = discover_papers(
    synonyms_path="data/synonyms.csv",
    categories=CATEGORY_MAP,
    sources=["openalex", "europepmc", "arxiv"],   # google_scholar opt-in
    download_pdfs=True,
    out_path="output/papers.json",
)
```

### 3.2 CLI

```
paper-discovery build-queries  --in synonyms.csv  --categories categories.json  --out queries.json
paper-discovery fetch          --queries queries.json --sources openalex,europepmc  --out papers.json
paper-discovery run            --in synonyms.csv  --categories categories.json  --out papers.json
```

### 3.3 HTTP service (optional wrapper — recommended if multiple consumer apps)

| Method | Path                    | Body / Query                                         | Returns                |
|--------|-------------------------|------------------------------------------------------|------------------------|
| POST   | `/v1/boolean-queries`   | `{ synonyms, categories }`                           | `CategoryQuery[]`      |
| POST   | `/v1/papers`            | `{ query, sources, max_pages?, download_pdfs? }`     | `Paper[]`              |
| POST   | `/v1/search`            | `{ synonyms, categories, sources, download_pdfs? }`  | `Paper[]` (full run)   |
| GET    | `/v1/sources`           | —                                                    | `SourceDescriptor[]`   |
| GET    | `/v1/healthz`           | —                                                    | `{status, sources}`    |

Long-running calls (`/v1/papers`, `/v1/search`) should be **async with a job-id**; clients poll `/v1/jobs/{id}`. A full multi-category run is minutes-to-hours.

---

## 4. Data contracts

### 4.1 Inputs

**Synonym table** — flat list of `{WORD, SYNONYMS}`. Origin format is open: CSV, Google Sheet export, JSON. The module needs only these two columns.

```jsonc
[
  { "WORD": "Foundation model",
    "SYNONYMS AND NEAR SYNONYMS": "LLM, large language model, GPT, foundation model" },
  { "WORD": "Humanitarian & Crisis Response",
    "SYNONYMS AND NEAR SYNONYMS": "humanitarian aid, disaster response, crisis intervention" }
]
```

> **Note.** The exact header `"SYNONYMS AND NEAR SYNONYMS"` is a CHITCHAT artifact; in a clean port, rename to `synonyms` and accept comma- *or* semicolon-separated values.

**Category map** — name → ordered list of `WORD` keys from the synonym table. Keys missing from the synonym table are dropped with a warning (do **not** fail the run).

```jsonc
{
  "Broad Foundational Search": [
      "Foundation model", "Artificial Intelligence System",
      "Machine Learning", "Deep Learning"
  ],
  "Humanitarian & Social Impact Search": [
      "Humanitarian & Crisis Response", "Core Humanitarian Principles", ...
  ]
}
```

### 4.2 Intermediate artifact — `CategoryQuery`

Produced by Stage 1, consumed by Stage 2. **This is the stable contract** between the two stages; keep it boring.

```jsonc
{
  "Combination_title":    "Humanitarian & Social Impact Search",
  "boolean_combination":  "(\"Foundation model\" OR \"LLM\") AND (\"humanitarian aid\" OR ...)"
}
```

### 4.3 Output — `Paper`

```jsonc
{
  "title":          "string | null",
  "authors":        "string, comma-separated | null",
  "url":            "string (landing page or PDF) | null",
  "abstract":       "string | null",
  "year":           "int | null",
  "extracted_text": "string (full PDF text) | null",
  "source":         "openalex | europepmc | arxiv | google_scholar",
  "search_category":"string (which CategoryQuery surfaced it)"
}
```

Fields are nullable on purpose — sources differ in what they return, and the consumer must tolerate gaps. **Do not invent defaults**; `null` means "we don't know."

---

## 5. Source adapters

Each adapter is a function with the signature:

```python
def search_<source>(query: str, **kwargs) -> list[Paper]: ...
```

| Source         | Endpoint                                            | Auth       | Pagination     | Rate-limit       | Notes |
|----------------|-----------------------------------------------------|------------|----------------|------------------|-------|
| OpenAlex       | `https://api.openalex.org/works`                    | none       | cursor (`*`)   | ~1 req/s polite  | Inverted-index abstracts must be rebuilt (`_rebuild_abstract`). PDF URL in `primary_location.pdf_url`. |
| Europe PMC     | `https://www.ebi.ac.uk/europepmc/webservices/rest/search` | none       | cursorMark     | ~1 req/s         | `resultType=core` is required for abstracts. Year may be under `journalInfo` or `bookOrReportDetails`. |
| arXiv          | `http://export.arxiv.org/api/query`                 | none       | `start`/`max_results` | **1 req / 3 s** (strict) | Atom XML. Boolean must be **simplified** to a few terms (see §6). |
| Google Scholar | via `scholarly` library                             | none, fragile | library-managed | aggressive throttling, captchas | Treat as best-effort, **opt-in**, may fail. |

### Adapter contract
- **Never raise on a remote error.** Log + return `[]`.
- **Always normalize** to the `Paper` schema before returning.
- **Skip PDF download** if `download_pdfs=False`. Adapters must respect this flag.
- **Failures are per-paper.** A broken PDF doesn't kill the rest of the result set.

---

## 6. Boolean query handling per source

OpenAlex and Europe PMC understand rich boolean expressions natively. **arXiv does not** — its query language is term-based and chokes on long `OR`-chains. The arXiv adapter therefore performs a **simplification step**:

1. Extract all double-quoted terms from the boolean string.
2. Take the top N (default 8) most-specific terms.
3. Emit a small set of `all:"<term>"` and `all:"<t1>" AND all:"<t2>"` queries.
4. Optionally add a category-tag query (`cat:cs.LG`, etc.) keyed off the `Combination_title`.

> **Implementation note.** The CHITCHAT code (`src/api/arxiv_paper_search.py`) hardcodes a `category_mappings` dict mapping category-title substrings to arXiv `cat:` codes. In a port, **lift this into config** (`arxiv_category_map.json`) so the consumer can customize without code changes.

Google Scholar accepts the raw boolean string but the `scholarly` library applies its own escaping. Don't pre-quote.

---

## 7. PDF download & text extraction

Two helpers, both lifted from `src/api/web_scrape.py`:

```python
download_research_paper(url: str, save_dir: str = "research_paper_downloads") -> str | None
extract_paper_text(pdf_path: str) -> str
```

`download_research_paper` is **deliberately generic**:
1. `GET` the URL with a `Mozilla/5.0` UA.
2. If `Content-Type` includes `pdf` or the URL ends in `.pdf`, stream to disk.
3. Otherwise parse HTML with BeautifulSoup, find the first `<a href>` whose href ends in `.pdf` / contains `/pdf` / `/epdf`, follow it, stream that.
4. Return path or `None`. **Never throws.**

`extract_paper_text` uses **PyPDF2** with two safety nets:
- Validates the `%PDF-` magic header.
- Pads a missing `%%EOF` marker (some publishers serve truncated files).
- Wraps in `BytesIO(strict=False)` to tolerate slightly malformed PDFs.

> **Carryover quirk.** The CHITCHAT requirements pin both `PyMuPDF` (`fitz`) and `PyPDF2`; only PyPDF2 is actually used in `extract_paper_text`. Drop the `fitz` import unless you intend to add OCR / image extraction.

PDFs are written to disk so that the consumer can also feed them to other text extractors (e.g. GROBID, unstructured.io) without re-downloading. **Keep this behavior** — it is load-bearing for downstream pipelines.

---

## 8. Dependencies (minimum viable port)

```
requests >= 2.32
beautifulsoup4 >= 4.13
PyPDF2 >= 3.0
scholarly >= 1.7        # only if google_scholar source enabled
pydantic >= 2.11        # for typed inputs/outputs (recommended)
```

**Drop from CHITCHAT requirements.txt when porting:** `fitz` / `PyMuPDF` (unused), `selenium`, `nipype`, `nibabel`, `sphinx*`, `bibtexparser`, `pandas` (Stage 2 doesn't need it), `openai` (that belongs to the screening module, not this one).

---

## 9. Suggested file layout in the consumer repo

```
paper_discovery/
├── __init__.py              # re-exports public API
├── boolean/
│   ├── __init__.py
│   ├── builder.py           # build_boolean_queries (was boolean_combinations.py)
│   └── composer.py          # category × AND composition (was unique_boolean_combinations.py)
├── sources/
│   ├── __init__.py
│   ├── base.py              # Source protocol / abstract adapter
│   ├── openalex.py
│   ├── europepmc.py
│   ├── arxiv.py             # contains the simplification step
│   └── google_scholar.py
├── pdf/
│   ├── __init__.py
│   ├── download.py          # download_research_paper
│   └── extract.py           # extract_paper_text
├── schema.py                # Pydantic models: Paper, CategoryQuery, SourceDescriptor
├── pipeline.py              # discover_papers (end-to-end orchestrator)
├── cli.py                   # paper-discovery CLI entrypoint
└── service/                 # optional HTTP wrapper
    ├── app.py               # FastAPI
    └── jobs.py              # background-task queue
```

---

## 10. Configuration knobs

Expose as constructor args / env vars — **do not hardcode**:

| Knob                     | Default | Notes |
|--------------------------|---------|-------|
| `RATE_LIMIT_OPENALEX`    | 1.0 s   | between pages |
| `RATE_LIMIT_EUROPEPMC`   | 1.0 s   | between pages |
| `RATE_LIMIT_ARXIV`       | 3.0 s   | between **queries** — arXiv ToS, do not lower |
| `MAX_PAGES_PER_QUERY`    | `None`  | `None` = paginate to end |
| `MAX_RESULTS_PER_QUERY`  | 200     | OpenAlex hard cap; Europe PMC up to 1000 |
| `PDF_DOWNLOAD_DIR`       | `./pdf_cache` | reused across runs — content-addressed names recommended |
| `USER_AGENT`             | `Mozilla/5.0` | be polite; set to `<your-app>/<version> (<email>)` |
| `ENABLE_GOOGLE_SCHOLAR`  | `False` | rate-limited, captcha-prone |
| `DEDUPE_KEY`             | `title.lower()` | replace with DOI when both available |

---

## 11. Errors, edge cases, things that bite

1. **arXiv 3-second rule.** Going below 3 s/req gets the IP banned for hours. The `time.sleep(3)` in the loop is not optional.
2. **Inverted-index abstracts.** OpenAlex returns abstracts as `{token: [positions]}` not strings. Use `_rebuild_abstract` *or* skip the abstract.
3. **Europe PMC paywalled PDFs.** The URL in `fullTextUrlList` may 200 with a paywall HTML page. The PDF extractor will fail; **catch and store `extracted_text=None`**, do not crash.
4. **Empty `key_terms` after simplification.** If a category has only multi-word terms, the arXiv `_extract_key_terms_from_boolean` filter (drops `or/and/the/for/with/data`) can return `[]`. Log + skip rather than firing a useless `all:"data"` query.
5. **Google Scholar captcha.** `scholarly` will silently start returning empty results. Detect zero-yield runs and surface a warning to the caller.
6. **PDF magic-byte check.** A file starting with `<!doctype html>` is the most common failure mode (HTTP 200 + HTML error page). The header check in `extract_paper_text` already handles it.
7. **Missing category keys.** If a category lists a key not in the synonym table, log a warning and continue with the keys that *are* present — never silently drop the whole category.
8. **Year coercion.** OpenAlex returns `int`, Europe PMC returns string, arXiv returns date-prefix. Normalize to `int | None`.

---

## 12. Extension points

Designed-in seams; the consumer codebase is expected to plug things in here:

- **New source adapter.** Implement the `Source` protocol (`search(query, **kw) -> list[Paper]`) and register it in `sources/__init__.py`. The pipeline picks it up by name.
- **Custom dedupe.** Replace the `DEDUPE_KEY` callable. Recommended: `(doi or normalize(title), year)`.
- **Post-fetch hooks.** Allow the consumer to pass `on_paper: Callable[[Paper], Paper]` for enrichment (DOI lookup, citation count, embedding, etc.).
- **Cache layer.** Stage 1 is pure; cache it by `hash(synonyms + categories)`. Stage 2 is expensive; cache by `hash(query + source)`.
- **Telemetry.** Emit one structured log per `(source, query)` pair: `{source, query, pages, papers, pdfs_attempted, pdfs_extracted, elapsed_s}`. Used to find dead repositories.

---

## 13. Integration checklist (for the assistant in the consumer repo)

- [ ] Copy `src/boolean/*.py` and `src/api/*.py` into `paper_discovery/` per the layout in §9.
- [ ] Rewrite the file-path-based helpers (`generate_boolean_combinations(input_file, output_file)`) into **pure functions** that take/return Python objects; keep the file-IO wrappers as thin CLI shims.
- [ ] Replace `print(...)` debug calls with `logging.getLogger(__name__)` and let the host app configure handlers.
- [ ] Add Pydantic models in `schema.py` matching §4.
- [ ] Lift the `predefined_combinations` dict out of `UniqueBooleanCombinationsGenerator.__init__` into a caller-provided argument (it is domain config, not library code).
- [ ] Lift the arXiv `category_mappings` dict into a JSON config file.
- [ ] Add a `Source` Protocol; convert each `search_*` function into an adapter class.
- [ ] Write one happy-path integration test per source, with mocked HTTP.
- [ ] Decide whether to ship the HTTP service wrapper (§3.3) or only the library (§3.1) — depends on how many apps need it.

---

## 14. Reference: source files in CHITCHAT

| Source file (CHITCHAT)                          | Maps to                          |
|-------------------------------------------------|----------------------------------|
| `src/boolean/boolean_combinations.py`           | `paper_discovery/boolean/builder.py` |
| `src/boolean/unique_boolean_combinations.py`    | `paper_discovery/boolean/composer.py` |
| `src/boolean/csv_to_json.py`                    | `paper_discovery/io.py` (synonym loader) |
| `src/api/web_scrape.py` → `search_openalex`     | `paper_discovery/sources/openalex.py` |
| `src/api/web_scrape.py` → `search_europepmc`    | `paper_discovery/sources/europepmc.py` |
| `src/api/web_scrape.py` → `search_google_scholar_scholarly` | `paper_discovery/sources/google_scholar.py` |
| `src/api/web_scrape.py` → `download_research_paper`, `extract_paper_text` | `paper_discovery/pdf/{download,extract}.py` |
| `src/api/arxiv_paper_search.py`                 | `paper_discovery/sources/arxiv.py` |

End of spec.
