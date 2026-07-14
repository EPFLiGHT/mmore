import cProfile
import os
import pstats
from typing import Mapping
from unittest.mock import MagicMock, patch

import pytest

from mmore.profiler import (
    Profiler,
    configure_profiling,
    enable_profiling_from_env,
    get_profiling_config,
    profile_context,
    profile_function,
    time_context,
    time_function,
)

StatsKey = tuple[str, int, str]
StatsValue = tuple[object, ...]


def _stats_map(stats: pstats.Stats) -> Mapping[StatsKey, StatsValue]:
    return getattr(stats, "stats")


@pytest.fixture(autouse=True)
def reset_profiler_config(tmp_path):
    """Reset global config after each test."""
    yield
    configure_profiling(enabled=False, output_dir=str(tmp_path))


# ------------------ Configuration Tests ------------------


def test_configure_and_get_profiling_config(tmp_path):
    """All fields round-trip correctly through configure/get."""
    configure_profiling(
        enabled=True,
        output_dir=str(tmp_path / "custom"),
        profile_functions=False,
        profile_memory=True,
        sort_by="time",
        max_results=10,
    )
    config = get_profiling_config()

    assert config.enabled is True
    assert config.output_dir == str(tmp_path / "custom")
    assert config.profile_functions is False
    assert config.profile_memory is True
    assert config.sort_by == "time"
    assert config.max_results == 10


def test_configure_profiling_creates_output_dir(tmp_path):
    """configure_profiling creates the output directory on disk."""
    out_dir = str(tmp_path / "new_dir")
    assert not os.path.exists(out_dir)
    configure_profiling(enabled=True, output_dir=out_dir)
    assert os.path.isdir(out_dir)


def test_enable_profiling_from_env(tmp_path, monkeypatch):
    """Parses all 4 env vars and updates global config accordingly."""
    env_dir = str(tmp_path / "env_dir")
    monkeypatch.setenv("MMORE_PROFILING_ENABLED", "true")
    monkeypatch.setenv("MMORE_PROFILING_OUTPUT_DIR", env_dir)
    monkeypatch.setenv("MMORE_PROFILING_SORT_BY", "calls")
    monkeypatch.setenv("MMORE_PROFILING_MAX_RESULTS", "25")
    enable_profiling_from_env()

    config = get_profiling_config()
    assert config.enabled is True
    assert config.output_dir == env_dir
    assert config.sort_by == "calls"
    assert config.max_results == 25
    assert os.path.isdir(env_dir)


def test_enable_profiling_from_env_defaults(tmp_path, monkeypatch):
    """Absent env vars result in default config values."""
    monkeypatch.chdir(tmp_path)
    for k in (
        "MMORE_PROFILING_ENABLED",
        "MMORE_PROFILING_OUTPUT_DIR",
        "MMORE_PROFILING_SORT_BY",
        "MMORE_PROFILING_MAX_RESULTS",
    ):
        monkeypatch.delenv(k, raising=False)

    enable_profiling_from_env()

    config = get_profiling_config()
    assert config.enabled is False
    assert config.output_dir == "./profiling_output"
    assert config.sort_by == "cumulative"
    assert config.max_results == 50
    assert os.path.isdir(tmp_path / "profiling_output")


# ------------------ Timing Utility Tests ------------------


@patch("mmore.profiler.logger")
def test_time_function_with_parens(mock_logger: MagicMock):
    """@time_function() with parens returns correct result and logs."""

    @time_function(log=True)
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    mock_logger.info.assert_called_once()
    assert "add took" in mock_logger.info.call_args[0][0]


@patch("mmore.profiler.logger")
def test_time_function_no_parens(mock_logger: MagicMock):
    """@time_function() returns correct result and logs."""

    @time_function()
    def multiply(a: int, b: int) -> int:
        return a * b

    assert multiply(3, 4) == 12
    mock_logger.info.assert_called_once()
    assert "multiply took" in mock_logger.info.call_args[0][0]


@patch("mmore.profiler.logger")
def test_time_context(mock_logger: MagicMock):
    """time_context logs elapsed time including block name."""
    with time_context("MyBlock", log=True):
        pass

    mock_logger.info.assert_called_once()
    assert "MyBlock took" in mock_logger.info.call_args[0][0]


@patch("mmore.profiler.logger")
def test_time_context_no_log(mock_logger: MagicMock):
    """time_context with log=False suppresses logger."""
    with time_context("SilentBlock", log=False):
        pass

    mock_logger.info.assert_not_called()


# ------------------ profile_function Tests ------------------


def test_profile_function_disabled(tmp_path):
    """When disabled, no .prof file is written and return value is correct."""
    configure_profiling(enabled=False, output_dir=str(tmp_path))

    @profile_function()
    def greet():
        return "hello"

    assert greet() == "hello"
    assert list(tmp_path.glob("*.prof")) == []


@patch("pstats.Stats.print_stats")
@patch("sys.getprofile", return_value=None)
def test_profile_function_enabled(_mock_getprofile, _mock_print_stats, tmp_path):
    """When enabled, a .prof file is written with func name in filename."""
    configure_profiling(enabled=True, output_dir=str(tmp_path))

    @profile_function()
    def compute():
        return 42

    assert compute() == 42
    files = list(tmp_path.glob("compute_*.prof"))
    assert len(files) == 1


@patch("pstats.Stats.print_stats")
@patch("sys.getprofile", return_value=None)
def test_profile_function_custom_output(_mock_getprofile, _mock_print_stats, tmp_path):
    """Explicit output_file param is used as the .prof path."""
    configure_profiling(enabled=True, output_dir=str(tmp_path))
    custom_path = str(tmp_path / "custom_output.prof")

    @profile_function(output_file=custom_path)
    def work():
        return "done"

    work()
    assert os.path.isfile(custom_path)


def test_profile_function_profile_functions_false(tmp_path):
    """enabled=True but profile_functions=False → no .prof file written."""
    configure_profiling(enabled=True, output_dir=str(tmp_path), profile_functions=False)

    @profile_function()
    def task():
        return "result"

    assert task() == "result"
    assert list(tmp_path.glob("*.prof")) == []


def test_profile_function_bypassed_if_profiler_active(tmp_path):
    """When sys.getprofile() returns non-None, profiling is skipped even if enabled."""
    configure_profiling(enabled=True, output_dir=str(tmp_path))

    @profile_function()
    def some_func():
        return "skipped"

    active_profiler_sentinel = object()
    with patch("sys.getprofile", return_value=active_profiler_sentinel):
        assert some_func() == "skipped"

    assert list(tmp_path.glob("*.prof")) == []


# ------------------ profile_context Tests ------------------


def test_profile_context_disabled(tmp_path):
    """When disabled, profile_context yields None and writes no file."""
    configure_profiling(enabled=False, output_dir=str(tmp_path))

    with profile_context("test_ctx") as prof:
        assert prof is None

    assert list(tmp_path.glob("*.prof")) == []


@patch("pstats.Stats.print_stats")
def test_profile_context_enabled(_mock_print_stats, tmp_path):
    """When enabled, profile_context yields a cProfile.Profile and writes a file."""
    configure_profiling(enabled=True, output_dir=str(tmp_path))

    with profile_context("test_ctx") as prof:
        assert isinstance(prof, cProfile.Profile)
        _ = sum(range(100))

    files = list(tmp_path.glob("test_ctx_*.prof"))
    assert len(files) == 1


# ------------------ Profiler Class Tests ------------------


@patch("pstats.Stats.print_stats")
def test_profiler_start_stop(_mock_print_stats, tmp_path):
    """Manual start/stop writes a named .prof file."""
    profiler = Profiler(enabled=True, output_dir=str(tmp_path))
    profiler.start()
    _ = [x**2 for x in range(100)]
    profiler.stop(name="manual_session")

    files = list(tmp_path.glob("manual_session_*.prof"))
    assert len(files) == 1


@patch("pstats.Stats.print_stats")
def test_profiler_context_manager(_mock_print_stats, tmp_path):
    """Context manager writes a file with the default 'session' name."""
    with Profiler(enabled=True, output_dir=str(tmp_path)) as prof:
        assert isinstance(prof.profiler, cProfile.Profile)
        _ = sum(range(100))

    files = list(tmp_path.glob("session_*.prof"))
    assert len(files) == 1


def test_profiler_disabled(tmp_path):
    """Profiler(enabled=False) — start/stop are no-ops and no file is written."""
    profiler = Profiler(enabled=False, output_dir=str(tmp_path))
    profiler.start()
    assert profiler.profiler is None
    profiler.stop(name="should_not_exist")

    assert list(tmp_path.glob("*.prof")) == []


def test_profiler_stop_without_start(tmp_path):
    """Calling stop() before start() is safe — no crash, no file written."""
    profiler = Profiler(enabled=True, output_dir=str(tmp_path))
    profiler.stop(name="premature_stop")

    assert list(tmp_path.glob("*.prof")) == []


# ------------------ Profile File Content Tests ------------------


@patch("pstats.Stats.print_stats")
@patch("sys.getprofile", return_value=None)
def test_profile_function_records_call_in_file(
    _mock_getprofile, _mock_print_stats, tmp_path
):
    """profile_function dumped file records the wrapped function call."""
    configure_profiling(enabled=True, output_dir=str(tmp_path))

    @profile_function()
    def workload():
        return sum(range(50))

    workload()

    files = list(tmp_path.glob("workload_*.prof"))
    assert len(files) == 1

    stats = pstats.Stats(str(files[0]))
    stats_map = _stats_map(stats)
    workload_entries = [key for key in stats_map if key[2] == "workload"]
    assert len(workload_entries) == 1
    assert stats_map[workload_entries[0]][1] == 1


@patch("pstats.Stats.print_stats")
def test_profile_context_records_total_call_count(_mock_print_stats, tmp_path):
    """profile_context records exact number of times an inner function is called."""
    configure_profiling(enabled=True, output_dir=str(tmp_path))

    def inner():
        return 1

    with profile_context("loop_ctx"):
        for _ in range(10):
            inner()

    files = list(tmp_path.glob("loop_ctx_*.prof"))
    assert len(files) == 1

    stats = pstats.Stats(str(files[0]))
    stats_map = _stats_map(stats)
    inner_entries = [key for key in stats_map if key[2] == "inner"]
    assert len(inner_entries) == 1
    assert stats_map[inner_entries[0]][1] == 10


@patch("pstats.Stats.print_stats")
def test_profiler_records_total_call_count(_mock_print_stats, tmp_path):
    """Profiler records exact number of times an inner function is called."""
    profiler = Profiler(enabled=True, output_dir=str(tmp_path))

    def inner():
        return 1

    profiler.start()
    for _ in range(10):
        inner()
    profiler.stop(name="loop_session")

    files = list(tmp_path.glob("loop_session_*.prof"))
    assert len(files) == 1

    stats = pstats.Stats(str(files[0]))
    stats_map = _stats_map(stats)
    inner_entries = [key for key in stats_map if key[2] == "inner"]
    assert len(inner_entries) == 1
    assert stats_map[inner_entries[0]][1] == 10
