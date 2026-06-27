"""End-to-end orchestrator: synonyms + categories -> deduplicated papers.json."""

import json
import logging
from pathlib import Path
from typing import List

from .boolean import build_boolean_queries, load_synonyms
from .config import PaperDiscoveryConfig
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
        queries = build_boolean_queries(synonyms, cfg.categories)
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
            adapter = get_adapter(
                src_name,
                user_agent=cfg.user_agent,
                max_pages=cfg.max_pages,
                max_results=cfg.max_results,
                **(
                    {"category_map": cfg.arxiv_category_map}
                    if src_name == "arxiv" and cfg.arxiv_category_map
                    else {}
                ),
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
                    paper.extracted_text = extract_text(str(cached_path)) or None
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
                paper.extracted_text = extract_text(result.path) or None
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
        out_path = Path(self.config.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = [p.to_dict() for p in papers]
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Wrote %d papers to %s", len(papers), out_path)


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
