"""End-to-end orchestrator: synonyms + categories -> deduplicated papers.json."""

import json
import logging
from pathlib import Path
from typing import List

import yaml
from dacite import from_dict

from .boolean import build_boolean_queries, load_synonyms
from .config import CategoriesFile, PaperDiscoveryConfig
from .pdf import download_pdf, expected_pdf_path, extract_text
from .schema import CategoryQuery, Paper
from .sources import get_adapter

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    # tqdm isn't in paper_discovery's hard deps - if missing, run silently
    # but still expose the methods we touch (set_postfix).
    class _NoTqdm:
        def __init__(self, iterable, **_kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, **_kwargs):
            pass

    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return _NoTqdm(iterable, **kwargs)


logger = logging.getLogger(__name__)


class PaperDiscoveryPipeline:
    """End-to-end Paper Discovery orchestrator.

    Drives the two pipeline stages off a single `PaperDiscoveryConfig`:
      1. Stage 1 (offline): build boolean queries from synonyms + categories.
      2. Stage 2 (online): fetch papers from each registered source, dedupe,
         and optionally download + extract text from PDFs.

    Results are written to `config.output_file` and also returned. Ctrl+C
    during stage 2 writes a partial `papers.json` before exiting.
    """

    def __init__(self, config: PaperDiscoveryConfig):
        self.config = config

    def run(self) -> List[Paper]:
        """Run the full pipeline and return the deduplicated `Paper` list.

        Side effect: writes `papers.json` to `config.output_file`. Safe to
        interrupt with Ctrl+C — partial results are saved.
        """
        cfg = self.config
        synonyms = load_synonyms(cfg.synonyms_path)
        categories = _load_categories(cfg.categories_path)
        queries = build_boolean_queries(synonyms, categories)
        logger.info("Built %d category queries", len(queries))

        all_papers: List[Paper] = []
        deduped: List[Paper] = []
        try:
            for q in queries:
                all_papers.extend(self._fetch_one(q))

            deduped = _dedupe(all_papers)
            logger.info(
                "After dedupe: %d papers (from %d)", len(deduped), len(all_papers)
            )

            if cfg.download_pdfs:
                self._enrich_with_pdf_text(deduped)
        except KeyboardInterrupt:
            deduped = _dedupe(all_papers) if not deduped else deduped
            logger.warning(
                "Interrupted - writing partial results (%d papers) to %s",
                len(deduped),
                cfg.output_file,
            )

        self._write_output(deduped)
        return deduped

    def _fetch_one(self, query: CategoryQuery) -> List[Paper]:
        cfg = self.config
        out: List[Paper] = []
        for src_name in cfg.sources:
            extra: dict = {}
            if src_name == "arxiv":
                if cfg.arxiv_category_map:
                    extra["category_map"] = cfg.arxiv_category_map
                extra["enable_pair_query"] = cfg.arxiv_enable_pair_query
            adapter = get_adapter(
                src_name,
                user_agent=cfg.user_agent,
                max_pages=cfg.max_pages,
                max_results=cfg.max_results,
                **extra,
            )
            logger.info("Searching %s for %r", src_name, query.combination_title)
            papers = adapter.search(query.boolean_combination, query.combination_title)
            logger.info("%s returned %d papers", src_name, len(papers))
            out.extend(papers)
        return out

    def _enrich_with_pdf_text(self, papers: List[Paper]) -> None:
        cfg = self.config
        cached = succeeded = paywalled = errored = skipped = 0
        bar = tqdm(papers, desc="PDFs", unit="paper")
        for paper in bar:
            if not paper.url:
                skipped += 1
                bar.set_postfix(
                    ok=succeeded, cache=cached, paywall=paywalled, err=errored
                )
                continue

            # Cache hit - skip the HTTP fetch entirely.
            if not cfg.force_redownload:
                cached_path = expected_pdf_path(paper.url, cfg.pdf_dir)
                if cached_path.exists():
                    paper.extracted_text = (
                        extract_text(str(cached_path), mode=cfg.pdf_extractor) or None
                    )
                    cached += 1
                    succeeded += 1
                    bar.set_postfix(
                        ok=succeeded, cache=cached, paywall=paywalled, err=errored
                    )
                    continue

            result = download_pdf(
                paper.url,
                cfg.pdf_dir,
                user_agent=cfg.user_agent,
                proxy_prefix=cfg.pdf_proxy_prefix,
            )
            if result.path:
                paper.extracted_text = (
                    extract_text(result.path, mode=cfg.pdf_extractor) or None
                )
                succeeded += 1
            elif result.paywalled:
                paywalled += 1
            elif result.errored:
                errored += 1
            else:
                skipped += 1
            bar.set_postfix(ok=succeeded, cache=cached, paywall=paywalled, err=errored)

        total = succeeded + paywalled + errored + skipped
        fresh = succeeded - cached
        logger.info(
            "PDF download: %d/%d succeeded (%d cached, %d fresh), "
            "%d paywalled, %d errors, %d skipped",
            succeeded,
            total,
            cached,
            fresh,
            paywalled,
            errored,
            skipped,
        )
        if paywalled and not cfg.pdf_proxy_prefix:
            logger.info(
                "Tip: %d PDFs were blocked by publisher paywalls. Set "
                "`pdf_proxy_prefix` in your config to use institutional "
                "access (e.g. EPFL: 'https://login.proxy.epfl.ch'), or "
                "set `download_pdfs: false` to skip PDFs entirely.",
                paywalled,
            )

    def _write_output(self, papers: List[Paper]) -> None:
        cfg = self.config
        out_path = Path(cfg.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for paper in papers:
                f.write(json.dumps(paper.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Wrote %d papers to %s", len(papers), out_path)

        if cfg.multimodal_output_file:
            self._write_multimodal_jsonl(papers, cfg.multimodal_output_file)

    def _write_multimodal_jsonl(self, papers: List[Paper], path: str) -> None:
        """Also emit results as JSONL of `MultimodalSample`.

        Feeds directly into mmore post-process / index / rag without a
        second processing pass.
        """
        from ..type import MultimodalSample

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Overwrite - `to_jsonl` appends by design, so start clean.
        if out_path.exists():
            out_path.unlink()

        samples = [
            p.to_multimodal_sample(
                pdf_path=str(expected_pdf_path(p.url, self.config.pdf_dir))
                if p.url and p.extracted_text
                else "",
            )
            for p in papers
        ]
        MultimodalSample.to_jsonl(str(out_path), samples)
        logger.info(
            "Wrote %d MultimodalSample records to %s (mmore-native shape)",
            len(samples),
            out_path,
        )


def _load_categories(path: str) -> dict:
    """Load `categories.yaml` via the `CategoriesFile` dataclass.

    The YAML file must hold a top-level `categories` key whose value is
    a mapping `{category_name -> [word, ...]}`. The dataclass wrapper
    gives us schema validation for free.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return from_dict(CategoriesFile, data).categories


def _dedupe(papers: List[Paper]) -> List[Paper]:
    seen = set()
    out: List[Paper] = []
    for p in papers:
        key = (p.title or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
