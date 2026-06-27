# mmore/run_paper_discovery.py
"""Entrypoint for `mmore paper-discovery --config-file <yaml>`."""

import argparse
import time

from dotenv import load_dotenv

from mmore.profiler import enable_profiling_from_env, profile_function

from .paper_discovery.config import PaperDiscoveryConfig
from .paper_discovery.logging_config import logger
from .paper_discovery.pipeline import PaperDiscoveryPipeline
from .utils import load_config

load_dotenv()


@profile_function()
def run_paper_discovery(config_file: str) -> None:
    cfg = load_config(config_file, PaperDiscoveryConfig)
    pipeline = PaperDiscoveryPipeline(config=cfg)
    logger.info("Running Paper Discovery pipeline...")
    start = time.time()
    pipeline.run()
    logger.info("Completed in %.2fs", time.time() - start)


if __name__ == "__main__":
    enable_profiling_from_env()
    parser = argparse.ArgumentParser(description="Run the Paper Discovery pipeline.")
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to the Paper Discovery configuration file (YAML).",
    )
    args = parser.parse_args()
    run_paper_discovery(args.config_file)
