# 📄 Paper Discovery

## Overview

The **Paper Discovery** module turns a list of domain keywords + synonyms into a
deduplicated set of academic papers (with metadata and extracted PDF text) by
federating queries across multiple academic repositories.

It has **no opinion about what comes next** — the output is a clean `Paper[]`
JSON list that any downstream consumer (indexing, RAG, screening) can use.

This page describes the standalone `paper_discovery` module. It is independent
of the `rag` and `index` pipelines.

## Installation

```bash
pip install "mmore[paper_discovery]"
```

For optional Google Scholar support (captcha-prone, best-effort):

```bash
pip install scholarly
```

`scholarly` is **not** in the `paper_discovery` extra by design — it is
captcha-prone. Install only if needed.

## Supported sources

| Source         | Auth | Notes |
|----------------|------|-------|
| OpenAlex       | none | ~1 req/s; inverted-index abstracts are rebuilt automatically |
| Europe PMC     | none | ~1 req/s; uses `resultType=core` |
| arXiv          | none | **1 req / 3 s** (strict, ToS); query is simplified to top terms; 30 s back-off on 429 |
| Google Scholar | none | Opt-in (`scholarly`), captcha-prone, best-effort |

## 🔁 Workflow

```
synonyms.json + categories
        │
        ▼
Stage 1: build boolean queries (pure, offline)
        │
        ▼
Stage 2: fetch from each source, dedupe, optionally download PDFs
        │
        ▼
   papers.json
```

Stage 1 is deterministic and offline. Stage 2 hits the network and is where all
rate limiting, retries, and PDF fetching live.

## 💻 Minimal Example

### 1. Prepare your synonym table

```json
[
  {"word": "Foundation model",
   "synonyms": ["LLM", "large language model", "GPT"]},
  {"word": "Humanitarian & Crisis Response",
   "synonyms": ["humanitarian aid", "disaster response"]}
]
```

Lookup of `word` from your `categories` config is **case-insensitive** —
`"Foundation model"`, `"foundation model"` and `"FOUNDATION MODEL"` all
match the same entry. Whitespace must still match exactly.

### 2. Create a config file

See [`examples/paper_discovery/config.yaml`](https://github.com/swiss-ai/mmore/blob/master/examples/paper_discovery/config.yaml).

### 3. Run the pipeline

```bash
python3 -m mmore paper-discovery --config-file examples/paper_discovery/config.yaml
```

Progress is shown live with a tqdm bar while PDFs are being downloaded:

```
PDFs:  42%|████▏     | 52/124 [01:15<01:43, 1.45s/paper, ok=42, cache=0, paywall=8, err=2]
```

Press **Ctrl+C** at any time — the pipeline catches the interrupt and writes
whatever it has so far to `output_file` before exiting.

## 📦 Output

A JSON array of `Paper` records:

```json
{
  "title": "A foundation model for humanitarian response",
  "authors": "Ada Lovelace, Alan Turing",
  "url": "https://arxiv.org/pdf/2401.00001.pdf",
  "abstract": "We introduce …",
  "year": 2024,
  "extracted_text": "<full PDF text>",
  "source": "arxiv",
  "search_category": "Humanitarian AI Search"
}
```

Fields are **nullable on purpose** — sources differ in what they return.
`null` means "we don't know."

## ⚙️ Configuration knobs

| Knob | Default | Notes |
|------|---------|-------|
| `sources` | `[openalex, europepmc, arxiv]` | Add `google_scholar` to opt in |
| `download_pdfs` | `true` | Set `false` to skip the PDF stage entirely |
| `max_pages` | `3` | Pages per source per query |
| `max_results` | `50` | Hard cap per source per query |
| `pdf_dir` | `./pdf_cache` | Reused across runs (see *PDF caching* below) |
| `force_redownload` | `false` | Set `true` to ignore the on-disk cache and re-fetch every PDF |
| `pdf_proxy_prefix` | `null` | Optional EZproxy prefix for institutional access (see *Paywalled PDFs* below) |
| `user_agent` | `mmore-paper-discovery/1.0 …` | HTTP `User-Agent` header sent on every outbound request — see below |
| `arxiv_category_map` | `null` | Maps a substring of your category title to an arXiv code (e.g. `Foundational` → `cs.LG`) — adds `cat:<code>` to the arXiv query |

### `user_agent`

The string set here is sent as the `User-Agent` HTTP header on every
request the pipeline makes — to OpenAlex, Europe PMC, arXiv, and to
publisher PDF endpoints. The default identifies mmore + the repo URL.

You should override it with a string that identifies your caller honestly
so rate-limiters / abuse desks can reach you. A concrete example:

```yaml
user_agent: "my-lab-pipeline/1.0 (mailto:alice@example.com)"
```

OpenAlex in particular routes UAs containing a contact address into a
faster, more reliable pool.

## 💾 PDF caching

`pdf_dir` is reused across runs. Before downloading a PDF, the pipeline checks
whether a file with the same name already exists; if so, the HTTP fetch is
skipped and text is extracted directly from the cached file.

The summary line at the end of a run shows the split:

```
PDF download: 108/124 succeeded (45 cached, 63 fresh), 16 paywalled, 0 errors, 0 skipped
```

This makes interrupted runs cheap to resume — every PDF that landed on disk
before Ctrl+C is reused, only the missing ones are fetched.

To force a full re-download (e.g. after a publisher updates a paper), set
`force_redownload: true` in your config.

## 🔒 Paywalled PDFs

Many publishers (Wiley, ACM, MDPI, …) block direct PDF downloads from
automated tools by design. **mmore does not disguise itself as a browser** —
that would violate publisher terms of service and risk getting the project's
default User-Agent blocklisted for every user. Two supported options:

### Institutional access via EZproxy (recommended)

Set `pdf_proxy_prefix` in your config to your institution's EZproxy URL. Every
paywalled URL will be wrapped through the proxy automatically:

```yaml
pdf_proxy_prefix: "https://login.proxy.epfl.ch"
```

The proxy handles SAML/Shibboleth authentication and the publisher sees a
valid institutional session.

**Caveat:** the first request through the proxy may redirect to your
institution's login page, which a script cannot fill in. The simplest
workaround for v1 is to sign in once in a browser to seed the session cookie,
then run the pipeline.

### Skip PDFs entirely

```yaml
download_pdfs: false
```

The pipeline still collects metadata + abstracts; only the `extracted_text`
field is left empty.

### Why we don't spoof the User-Agent

A common workaround for publisher 403s is to set the `User-Agent` to a
browser string (Chrome, Firefox, …). mmore does **not** do that by
default for two reasons:

1. It violates most publishers' terms of service.
2. A baked-in spoofed UA gets the **library's** default identifier
   blocklisted on first abuse — for every downstream user.

If you have a specific arrangement with a publisher (e.g. a registered
crawler agreement), you can set `user_agent` to whatever they require.
That's an opt-in you take responsibility for — not a default the library
ships.

## 🐍 Programmatic use

For embedding the pipeline in another script:

```python
from mmore.paper_discovery import PaperDiscoveryConfig, PaperDiscoveryPipeline
from mmore.utils import load_config

config = load_config("examples/paper_discovery/config.yaml", PaperDiscoveryConfig)
papers = PaperDiscoveryPipeline(config).run()
print(f"Got {len(papers)} papers")
```

Or compose Stage 1 alone (no network) for testing:

```python
from mmore.paper_discovery import build_boolean_queries
from mmore.paper_discovery.boolean import load_synonyms

synonyms = load_synonyms("examples/paper_discovery/synonyms.json")
queries = build_boolean_queries(synonyms, {"My Category": ["Foundation model"]})
for q in queries:
    print(q.combination_title, "->", q.boolean_combination)
```

## See also

- [Indexing](../getting_started/indexing.md) — feed `extracted_text` into the indexer
- [RAG](../getting_started/rag.md) — query the indexed papers
- [Processing pipeline](../getting_started/process.md) — convert other document formats
