import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from mmore.process.crawler import Crawler, CrawlerConfig
from mmore.process.dispatcher import Dispatcher, DispatcherConfig
from mmore.process.drive_download import GoogleDriveDownloader
from mmore.process.incremental import (
    is_reusable_process,
    load_previous_process_results,
)
from mmore.profiler import enable_profiling_from_env, profile_function
from mmore.utils import load_config
from mmore.ux import (
    PROCESS_EMOJI,
    PROCESS_NAME,
    human_size,
    quiet_noisy_libs,
    setup_logging,
    step_intro,
    step_summary,
)

logger = setup_logging(PROCESS_NAME, PROCESS_EMOJI)

overall_start_time = time.time()

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


@dataclass
class ProcessInference:
    """Inference configuration."""

    data_path: Union[List[str], str]
    google_drive_ids: List[str]
    dispatcher_config: DispatcherConfig
    previous_results: Optional[str] = None


def merged_results_path(output_path: str) -> str:
    """Path where `process` writes its final merged JSONL.

    Single source of truth for downstream tooling (TUI, scripts) that needs
    to locate the JSONL produced by a `process` run from its config.
    """
    return os.path.join(output_path, "merged", "merged_results.jsonl")


def _write_merged_results(output_path, reused_samples, dispatched=True) -> int:
    """Merge per-processor JSONL files and reused samples into a single output and
    returns the total number of samples written."""
    output_file = merged_results_path(output_path)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total_results = 0
    with open(output_file, "w") as f:
        for sample in reused_samples:
            f.write(json.dumps(sample.to_dict()) + "\n")
            total_results += 1
        if dispatched:
            processors_dir = os.path.join(output_path, "processors")
            if os.path.isdir(processors_dir):
                for processor_name in sorted(os.listdir(processors_dir)):
                    results_path = os.path.join(
                        processors_dir, processor_name, "results.jsonl"
                    )
                    if os.path.exists(results_path):
                        with open(results_path, "r") as processor_file:
                            for line in processor_file:
                                stripped_line = line.strip()
                                if stripped_line:
                                    f.write(stripped_line + "\n")
                                    total_results += 1

    logger.debug(f"Merged results ({total_results} samples) saved to {output_file}")
    return total_results


@profile_function()
def process(config_file: str):
    """Process documents from a directory."""
    quiet_noisy_libs()
    logger.debug(f"Dispatcher configuration file path: {config_file}")

    overall_start_time = time.time()

    config: ProcessInference = load_config(config_file, ProcessInference)

    ggdrive_downloader, ggdrive_download_dir = None, None
    if config.google_drive_ids:
        google_drive_ids = config.google_drive_ids
        ggdrive_downloader = GoogleDriveDownloader(google_drive_ids)
        ggdrive_downloader.download_all()
        ggdrive_download_dir = ggdrive_downloader.download_dir

    data_path = config.data_path or ggdrive_download_dir

    if data_path:
        if isinstance(data_path, str):
            data_path = [data_path]

        # add the ggdrive_download_dir only if needed
        if config.data_path and ggdrive_download_dir:
            data_path += [ggdrive_download_dir]

        crawler_config = CrawlerConfig(
            root_dirs=data_path,
            supported_extensions=[
                ".pdf",
                ".docx",
                ".pptx",
                ".md",
                ".txt",  # Document files
                ".xlsx",
                ".xls",
                ".csv",  # Spreadsheet files
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",  # Video files
                ".mp3",
                ".wav",
                ".aac",  # Audio files
                ".eml",  # Emails
                ".html",
                ".htm",  # HTML pages
            ],
            output_path=config.dispatcher_config.output_path,
        )
    else:
        raise ValueError("Data path not provided in the configuration")

    logger.debug(f"Using crawler configuration: {crawler_config}")
    crawler = Crawler(config=crawler_config)

    crawl_start_time = time.time()
    crawl_result = crawler.crawl()
    crawl_end_time = time.time()
    crawl_time = crawl_end_time - crawl_start_time
    logger.debug(f"Crawled {len(crawl_result)} files in {crawl_time:.4f}s")

    # Collect all crawled file paths and urls (excluding this way deleted files)
    all_crawled_paths = {
        fd.file_path
        for file_list in crawl_result.file_paths.values()
        for fd in file_list
    }
    all_crawled_paths.update(url.file_path for url in crawl_result.urls)

    previous = None
    reused_samples = []
    reusable_paths = set()

    if config.previous_results:
        previous = load_previous_process_results(config.previous_results)

        for fp in all_crawled_paths:
            if is_reusable_process(fp, previous):
                reusable_paths.add(fp)

        reused_samples = [previous[fp] for fp in sorted(reusable_paths)]

        # Remove reusable files from crawl_result so they are not re-processed
        crawl_result.file_paths = {
            root_dir: [fd for fd in file_list if fd.file_path not in reusable_paths]
            for root_dir, file_list in crawl_result.file_paths.items()
        }

        n_deleted = len(set(previous.keys()) - all_crawled_paths)
        logger.info(
            f"Process pipeline: {len(reusable_paths)} reused, "
            f"{len(crawl_result)} to process, {n_deleted} deleted"
        )

    output_path = config.dispatcher_config.output_path

    dispatched = len(crawl_result) > 0
    n_dispatched = len(crawl_result)

    intro_fields = [f"{len(all_crawled_paths)} files", f"output: {output_path}"]
    step_intro(
        PROCESS_NAME,
        PROCESS_EMOJI,
        "Read your files and pull out their text and images",
        intro_fields,
    )

    if not dispatched and not reused_samples:
        logger.warning("⚠️ Found no file to process")
        if previous is None:
            return

    dispatch_time = 0.0
    if dispatched:
        dispatcher_config: DispatcherConfig = config.dispatcher_config
        logger.debug(f"Using dispatcher configuration: {dispatcher_config}")

        dispatcher = Dispatcher(result=crawl_result, config=dispatcher_config)
        dispatch_start_time = time.time()
        list(dispatcher())
        dispatch_time = time.time() - dispatch_start_time
        logger.debug(f"Processed {n_dispatched} files in {dispatch_time:.1f}s")
    elif reused_samples:
        logger.debug("No new files to process, reusing previous samples only.")
    else:
        logger.debug("No new files to process and no samples to reuse.")

    total_samples = _write_merged_results(
        output_path,
        reused_samples,
        dispatched=dispatched,
    )

    if ggdrive_downloader:
        ggdrive_downloader.remove_downloads()

    overall_time = time.time() - overall_start_time
    merged_file = merged_results_path(output_path)
    size_label = (
        human_size(os.path.getsize(merged_file)) if os.path.exists(merged_file) else "-"
    )
    stats: dict[str, object] = {}
    if reused_samples:
        stats["files"] = f"{n_dispatched} new · {len(reused_samples)} reused ♻️"
    stats["output"] = f"{total_samples} samples"
    stats["size"] = size_label
    stats["throughput"] = (
        f"{n_dispatched / dispatch_time:.2f} files/s" if dispatch_time else "-"
    )
    step_summary(PROCESS_NAME, PROCESS_EMOJI, overall_time, stats)


if __name__ == "__main__":
    enable_profiling_from_env()
    parser = argparse.ArgumentParser(description="Run the processing pipeline.")
    parser.add_argument(
        "--config_file", required=True, help="Path to the process configuration file."
    )
    args = parser.parse_args()

    process(args.config_file)
