# 📄 Paper Discovery

## Overview

The **Paper Discovery** module helps you build a targeted collection of academic papers on a topic you care about. You describe the topic once — as a list of keywords with synonyms — and the module searches several open academic repositories (OpenAlex, Europe PMC, arXiv, optionally Google Scholar) on your behalf, downloads whatever PDFs it can, and writes a single JSON file with the metadata and extracted text.

The output is a plain `Paper[]` JSON list. What you do with it next is up to you — feed it into an indexer, hand it to a screening tool, or just read the abstracts. This page describes the standalone `paper_discovery` module. It is independent of the `rag` and `index` pipelines.

## Installation

```bash
uv pip install "mmore[paper_discovery]"
```

For optional Google Scholar support (captcha-prone, best-effort):

```bash
uv pip install scholarly
```

`scholarly` is **not** in the `paper_discovery` extra by design — it is captcha-prone. Install only if needed.

## Supported sources

| Source | What it covers |
|--------|----------------|
| **OpenAlex** | Broadest general index of academic papers. Abstracts included by default. |
| **Europe PMC** | Biomedical and life-sciences literature with links to full text where available. |
| **arXiv** | Preprints in ML, physics, math, and CS. Slower than the others because arXiv enforces a 3-second gap between requests. |
| **Google Scholar** | Widest overall coverage but captcha-prone. Opt-in — requires `scholarly`. |

All four sources are anonymous — no API keys needed. Precise rate limits, retry back-off, and API-specific details live in each adapter's docstring under `src/mmore/paper_discovery/sources/`.

## 🔁 Workflow

```
synonyms.jsonl + categories.yaml
        │
        ▼
Stage 1: build boolean queries (pure, offline)
        │
        ▼
Stage 2: fetch from each source, dedupe, optionally download PDFs
        │
        ▼
   papers.jsonl
```

Stage 1 doesn't touch the network — it just turns your synonyms + categories into search queries. Stage 2 is where everything network-related happens: hitting each source, respecting their rate limits, downloading PDFs, retrying when things go wrong.

## 💻 Minimal Example

### 1. Prepare your synonym table

A **JSONL** file with one `{"word": ..., "synonyms": [...]}` object per line. Easy to diff, append, and edit line-by-line:

```jsonl
{"word": "Foundation model", "synonyms": ["LLM", "large language model", "GPT"]}
{"word": "Humanitarian & Crisis Response", "synonyms": ["humanitarian aid", "disaster response"]}
```

You don't have to worry about capitalization — `"Foundation model"`, `"foundation model"` and `"FOUNDATION MODEL"` are treated as the same word. Whitespace does need to match. If any of your terms happen to contain a `"` character, don't stress — it's silently stripped when the file is loaded.

### 2. Define your categories

Categories live in their own YAML file, loaded via a small `CategoriesFile` dataclass:

```yaml
# categories.yaml
categories:
  Broad Foundational Search:
    - Foundation model
    - Machine Learning
  Humanitarian AI Search:
    - Foundation model
    - Humanitarian & Crisis Response
```

Each name under a category must match a `word` in your synonyms file. For every category, the module builds one search that finds papers mentioning **at least one term from each group of synonyms**. So the "Broad Foundational Search" example above will match a paper if it talks about *any* foundation-model synonym AND *any* machine-learning synonym.

### 3. Create a config file

See [`examples/paper_discovery/config.yaml`](https://github.com/EPFLiGHT/mmore/blob/master/examples/paper_discovery/config.yaml). It points at your `synonyms_path` and `categories_path`.

### 4. Run the pipeline

```bash
python3 -m mmore paper-discovery --config-file examples/paper_discovery/config.yaml
```

Progress is shown live with a tqdm bar while PDFs are being downloaded:

```
PDFs:  42%|████▏     | 52/124 [01:15<01:43, 1.45s/paper, ok=42, cache=0, paywall=8, err=2]
```

Press **Ctrl+C** at any time — the pipeline catches the interrupt and writes whatever it has so far to `output_file` before exiting.

## 📦 Output

A JSONL file — one `Paper` record per line. Example line:

```json
{"title": "A foundation model for humanitarian response", "authors": ["Ada Lovelace", "Alan Turing"], "url": "https://arxiv.org/pdf/2401.00001.pdf", "abstract": "We introduce …", "year": 2024, "extracted_text": "<full PDF text>", "source": "arxiv", "search_category": "Humanitarian AI Search"}
```

Line-per-record makes the file streamable (read one paper at a time), diff-friendly, and easy to append to. Any JSONL-aware tool (`jq`, `pandas.read_json(lines=True)`, mmore's `MultimodalSample.from_jsonl`) can consume it directly.

Fields are **nullable on purpose** — sources differ in what they return. `null` means "we don't know."

## ⚙️ Configuration knobs

| Knob | Default | Notes |
|------|---------|-------|
| `synonyms_path` | *(required)* | Path to a `.jsonl` synonyms file (one object per line) |
| `categories_path` | *(required)* | Path to a `categories.yaml` file (see step 2) |
| `sources` | `[openalex, europepmc, arxiv]` | Add `google_scholar` to opt in |
| `download_pdfs` | `true` | Set `false` to skip the PDF stage entirely |
| `max_pages` | `3` | Pages per source per query |
| `max_results` | `50` | Hard cap per source per query |
| `pdf_dir` | `./pdf_cache` | Reused across runs (see *PDF caching* below) |
| `force_redownload` | `false` | Set `true` to ignore the on-disk cache and re-fetch every PDF |
| `pdf_extractor` | `"fast"` | Which mmore PDF processor to use. `"fast"` = PyMuPDF-backed, no models loaded. `"full"` = marker + surya for better parsing (slow, downloads models) |
| `multimodal_output_file` | `null` | If set, also write a JSONL of `MultimodalSample` records that mmore's post-process / index / RAG pipelines can consume directly (see [Feeding results into mmore's index / RAG](#-feeding-results-into-mmores-index--rag)) |
| `pdf_proxy_prefix` | `null` | EZproxy host, only if your institution runs one. Leave unset for VPN-based access (see *Paywalled PDFs* below) |
| `user_agent` | `mmore-paper-discovery/1.0 …` | HTTP `User-Agent` header sent on every outbound request — see below |
| `arxiv_category_map` | `null` | Maps a substring of your category title to an arXiv code (e.g. `Foundational` → `cs.LG`) — adds `cat:<code>` to the arXiv query |
| `arxiv_enable_pair_query` | `true` | Runs one extra arXiv search per category that requires the top two terms together (better precision). Turn off if you'd rather save a few seconds per category |

### `user_agent`

This is the "who's asking?" string sent with every network request the module makes. Sources use it to identify who's hitting their API, and OpenAlex specifically gives faster, more reliable responses to requests that include a contact address.

You should set it to something that identifies your project so the source's team can reach you if you're making too many requests. A concrete example:

```yaml
user_agent: "my-lab-pipeline/1.0 (mailto:alice@example.com)"
```

The default just identifies mmore + the repo URL, which works but doesn't tell anyone who *you* are.

## 💾 PDF caching

`pdf_dir` is reused across runs. Before downloading a PDF, the pipeline checks whether a file with the same name already exists; if so, the HTTP fetch is skipped and text is extracted directly from the cached file.

The summary line at the end of a run shows the split:

```
PDF download: 108/124 succeeded (45 cached, 63 fresh), 16 paywalled, 0 errors, 0 skipped
```

This makes interrupted runs cheap to resume — every PDF that landed on disk before Ctrl+C is reused, only the missing ones are fetched.

To force a full re-download (e.g. after a publisher updates a paper), set `force_redownload: true` in your config.

## 🔒 Paywalled PDFs

Expect a chunk of your run to come back without full text. Some of that you can fix, some of it you cannot. Read this section before spending time on it.

### There are two different reasons a PDF fails

They look the same in the summary line but have completely different fixes.

**1. You don't have access.** Your institution has no subscription to that journal. Nothing in this pipeline can fix that. Request the paper through your library instead.

**2. You have access, but the publisher blocks automated tools.** This is the common one, and it surprises people. Publishers like Wiley, ACM, and Science return `403` to anything that doesn't look like a browser, *regardless of whether your institution subscribes*. You can click the same link in your browser and get the PDF, then watch the pipeline get refused for the identical URL.

mmore does **not** work around this by pretending to be a browser. Spoofing the User-Agent violates most publishers' terms of service, and a spoofed default would get the project's identifier blocklisted for every user of the library. That's a deliberate choice, not an oversight.

### How your institution grants access matters

Two common models. Check which one yours uses before touching any config.

**VPN and IP recognition.** You connect to your institution's VPN, the publisher sees an institutional IP, and access is granted automatically. **Leave `pdf_proxy_prefix` unset.** The direct URL already works. EPFL works this way.

**EZproxy.** Your library gives you a hostname that rewrites URLs. If, and only if, your institution runs one, set:

```yaml
pdf_proxy_prefix: "https://ezproxy.example.edu"
```

Use the exact host your library publishes. Do not guess it. A wrong host either fails DNS or serves you a login page, and neither yields a PDF.

Even with the right host, an EZproxy that needs an interactive sign-in will return its login page instead of the PDF. The pipeline detects this and warns you:

```
12 downloads returned a sign-in page instead of a PDF. This pipeline
cannot log in for you.
```

There is no headless workaround for that today. The pipeline cannot complete a SAML or Shibboleth login.

### Skip PDFs entirely

If full text isn't essential, this is the cheapest path and it always works:

```yaml
download_pdfs: false
```

You still get every paper's metadata and abstract. Only `extracted_text` is left empty.

## 📄 PDF text extraction

Text extraction goes through the same PDF processor the rest of mmore uses, so you get consistent output whether a paper comes from Paper Discovery or from another `mmore process` run. There are two settings you can pick between with `pdf_extractor`:

- **`fast` (default)** — Uses PyMuPDF under the hood. Nothing to download, works right out of the box, and it's good enough for most academic PDFs.
- **`full`** — Uses mmore's fuller pipeline (with layout-aware parsing). Better on messy PDFs — multi-column layouts, scanned pages, complex figures — but it downloads model weights the first time it runs, and it's really only worth it if you have a GPU.

Start with `fast`. Only switch to `full` if you notice extraction is losing structure on the papers you care about.

### Why we don't spoof the User-Agent

A common workaround for publisher 403s is to set the `User-Agent` to a browser string (Chrome, Firefox, …). mmore does **not** do that by default for two reasons:

1. It violates most publishers' terms of service.
2. A baked-in spoofed UA gets the **library's** default identifier blocklisted on first abuse — for every downstream user.

If you have a specific arrangement with a publisher (e.g. a registered crawler agreement), you can set `user_agent` to whatever they require. That's an opt-in you take responsibility for — not a default the library ships.

## 🔌 Feeding results into mmore's index / RAG

If you plan to index the discovered papers or run RAG over them, you don't need to send them back through `mmore process`. Ask the pipeline to write an extra output file in mmore's canonical `MultimodalSample` shape:

```yaml
multimodal_output_file: examples/paper_discovery/papers.samples.jsonl
```

Every paper is converted to a `MultimodalSample`:

- **`text`** — the extracted PDF body if we downloaded it, otherwise the abstract, otherwise the title.
- **`metadata.file_path`** — points at the cached PDF when we have one.
- **`metadata.processor_type`** — always `"paper_discovery"`, so downstream filters can recognise the source.
- **`metadata.extra`** — carries the paper-specific fields (title, authors, year, source, url, search_category, abstract).

The resulting JSONL is a drop-in input for the post-process, index, and RAG pipelines. The default `papers.jsonl` output is still written the same way alongside it.

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

synonyms = load_synonyms("examples/paper_discovery/synonyms.jsonl")
queries = build_boolean_queries(synonyms, {"My Category": ["Foundation model"]})
for q in queries:
    print(q.combination_title, "->", q.boolean_combination)
```

## See also

- [Indexing](../getting_started/indexing.md) — feed `extracted_text` into the indexer
- [RAG](../getting_started/rag.md) — query the indexed papers
- [Processing pipeline](../getting_started/process.md) — convert other document formats
