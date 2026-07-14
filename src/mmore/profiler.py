"""Profiling utilities for performance monitoring and analysis."""

import cProfile
import functools
import logging
import os
import pstats
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

R = TypeVar("R")


@dataclass
class ProfilingConfig:
    """Configuration for profiling."""

    enabled: bool = False
    output_dir: str = "./profiling_output"
    profile_functions: bool = True
    profile_memory: bool = False
    sort_by: str = "cumulative"  # cumulative, time, calls, etc.
    max_results: int = 50


# Global profiling configuration
_profiling_config = ProfilingConfig()


def configure_profiling(
    enabled: bool = True,
    output_dir: str = "./profiling_output",
    profile_functions: bool = True,
    profile_memory: bool = False,
    sort_by: str = "cumulative",
    max_results: int = 50,
):
    """Configure global profiling settings.

    Args:
        enabled: Enable or disable profiling globally
        output_dir: Directory to save profiling results
        profile_functions: Enable function-level profiling
        profile_memory: Enable memory profiling (requires memory_profiler)
        sort_by: How to sort profiling results (cumulative, time, calls, etc.)
        max_results: Maximum number of results to show
    """
    global _profiling_config
    _profiling_config = ProfilingConfig(
        enabled=enabled,
        output_dir=output_dir,
        profile_functions=profile_functions,
        profile_memory=profile_memory,
        sort_by=sort_by,
        max_results=max_results,
    )
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Profiling configured: enabled={enabled}, output_dir={output_dir}")


def get_profiling_config() -> ProfilingConfig:
    """Get the current profiling configuration."""
    return _profiling_config


def profile_function(
    output_file: Optional[str] = None,
    sort_by: Optional[str] = None,
    max_results: Optional[int] = None,
):
    """Decorator to profile a function.

    Args:
        output_file: Optional file path to save profiling results
        sort_by: How to sort results (overrides global config)
        max_results: Maximum results to show (overrides global config)

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            config = get_profiling_config()
            if (
                not config.enabled
                or not config.profile_functions
                or sys.getprofile() is not None
            ):
                return func(*args, **kwargs)

            # Generate output file name if not provided
            if output_file is None:
                func_name = func.__name__
                timestamp = time.time_ns()
                output_path = os.path.join(
                    config.output_dir, f"{func_name}_{timestamp}.prof"
                )
            else:
                output_path = output_file

            # Create profiler
            profiler = cProfile.Profile()
            profiler.enable()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.disable()

                # Save profiling results
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                profiler.dump_stats(output_path)

                # Print summary
                stats = pstats.Stats(profiler)
                stats.sort_stats(sort_by or config.sort_by)
                stats.print_stats(max_results or config.max_results)

                logger.info(f"Profiling results saved to: {output_path}")

        return wrapper

    return decorator


@contextmanager
def profile_context(
    name: str,
    output_file: Optional[str] = None,
    sort_by: Optional[str] = None,
    max_results: Optional[int] = None,
):
    """Context manager for profiling a code block.

    Args:
        name: Name identifier for this profiling session
        output_file: Optional file path to save profiling results
        sort_by: How to sort results (overrides global config)
        max_results: Maximum results to show (overrides global config)

    Yields:
        Profiler instance
    """
    config = get_profiling_config()
    if not config.enabled or not config.profile_functions:
        yield None
        return

    # Generate output file name if not provided
    if output_file is None:
        timestamp = time.time_ns()
        output_path = os.path.join(config.output_dir, f"{name}_{timestamp}.prof")
    else:
        output_path = output_file

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        yield profiler
    finally:
        profiler.disable()

        # Save profiling results
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        profiler.dump_stats(output_path)

        # Print summary
        stats = pstats.Stats(profiler)
        stats.sort_stats(sort_by or config.sort_by)
        stats.print_stats(max_results or config.max_results)

        logger.info(f"Profiling results saved to: {output_path}")


@contextmanager
def time_context(name: str, log: bool = True):
    """Context manager for timing a code block.

    Args:
        name: Name identifier for this timing session
        log: Whether to log the timing result

    Yields:
        None
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if log:
            logger.info(f"⏱️  {name} took {elapsed:.4f} seconds")


def time_function(log: bool = True) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Decorator to time a function execution.

    Args:
        log: Whether to log the timing result

    Returns:
        Decorated function
    """

    def decorator(f: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(f)
        def wrapper(*args: object, **kwargs: object) -> R:
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start_time
                if log:
                    logger.info(f"⏱️  {f.__name__} took {elapsed:.4f} seconds")

        return wrapper

    return decorator


class Profiler:
    """Main profiler class for managing profiling sessions."""

    def __init__(
        self,
        enabled: bool = True,
        output_dir: str = "./profiling_output",
        profile_functions: bool = True,
        profile_memory: bool = False,
    ):
        """Initialize profiler.

        Args:
            enabled: Enable profiling
            output_dir: Directory to save profiling results
            profile_functions: Enable function-level profiling
            profile_memory: Enable memory profiling
        """
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.profile_functions = profile_functions
        self.profile_memory = profile_memory
        self.profiler: Optional[cProfile.Profile] = None
        self.start_time: Optional[float] = None

        if enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Profiler initialized: output_dir={output_dir}")

    def start(self):
        """Start profiling."""
        if not self.enabled or not self.profile_functions:
            return

        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.start_time = time.time()
        logger.info("Profiling started")

    def stop(
        self, name: str = "session", sort_by: str = "cumulative", max_results: int = 50
    ):
        """Stop profiling and save results.

        Args:
            name: Name for the profiling session
            sort_by: How to sort results
            max_results: Maximum number of results to show
        """
        if not self.enabled or not self.profile_functions or self.profiler is None:
            return

        self.profiler.disable()

        # Generate output file name
        timestamp = time.time_ns()
        output_path = self.output_dir / f"{name}_{timestamp}.prof"

        # Save profiling results
        self.profiler.dump_stats(str(output_path))

        # Print summary
        stats = pstats.Stats(self.profiler)
        stats.sort_stats(sort_by)
        stats.print_stats(max_results)

        elapsed = time.time() - (self.start_time or time.time())
        logger.info(f"Profiling stopped: {elapsed:.4f} seconds")
        logger.info(f"Profiling results saved to: {output_path}")

        self.profiler = None
        self.start_time = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def enable_profiling_from_env():
    """Enable profiling from environment variables.

    Environment variables:
        MMORE_PROFILING_ENABLED: Enable profiling (default: false)
        MMORE_PROFILING_OUTPUT_DIR: Output directory (default: ./profiling_output)
        MMORE_PROFILING_SORT_BY: Sort results by (default: cumulative)
        MMORE_PROFILING_MAX_RESULTS: Max results to show (default: 50)
    """
    enabled = os.getenv("MMORE_PROFILING_ENABLED", "false").lower() == "true"
    output_dir = os.getenv("MMORE_PROFILING_OUTPUT_DIR", "./profiling_output")
    sort_by = os.getenv("MMORE_PROFILING_SORT_BY", "cumulative")
    max_results = int(os.getenv("MMORE_PROFILING_MAX_RESULTS", "50"))

    configure_profiling(
        enabled=enabled,
        output_dir=output_dir,
        sort_by=sort_by,
        max_results=max_results,
    )
