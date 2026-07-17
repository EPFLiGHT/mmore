"""Unit tests for the shared privacy model registry."""

from typing import Callable

import pytest
from presidio_analyzer import AnalyzerEngine

from mmore.privacy import _cache
from mmore.privacy._cache import ModelRegistry

MB = 1024 * 1024


class _TestModel(AnalyzerEngine):
    def __init__(self, tag: str) -> None:
        self.tag = tag


class _FakeMem:
    """Stand-in for device memory: loaders bump ``used``, the registry reads it."""

    def __init__(self) -> None:
        self.used = 0

    def __call__(self) -> int:
        return self.used

    def loader(self, size: int, tag: str) -> Callable[[], _TestModel]:
        def load() -> _TestModel:
            self.used += size
            return _TestModel(tag)

        return load


def _fail() -> _TestModel:
    raise AssertionError("loader should not run for a cached key")


@pytest.fixture
def fake_mem(monkeypatch):
    mem = _FakeMem()
    monkeypatch.setattr(_cache, "_device_mem_bytes", mem)
    monkeypatch.setattr(_cache, "_empty_device_cache", lambda: None)
    return mem


def test_loads_once_and_reuses():
    reg = ModelRegistry(budget_mb=0)
    assert reg.get_or_load("k", lambda: _TestModel("V")).tag == "V"
    assert reg.get_or_load("k", _fail).tag == "V"


def test_zero_budget_disables_eviction():
    reg = ModelRegistry(budget_mb=0)
    for i in range(5):
        reg.get_or_load(f"k{i}", lambda i=i: _TestModel(f"V{i}"))
    for i in range(5):
        assert reg.get_or_load(f"k{i}", _fail).tag == f"V{i}"


def test_auto_budget_uses_device_total(monkeypatch, fake_mem):
    monkeypatch.setattr(_cache, "_BUDGET_FRACTION", 1.0)
    monkeypatch.setattr(_cache, "_device_total_bytes", lambda: 25 * MB)
    reg = ModelRegistry()  # no budget -> auto-detect
    reg.get_or_load("a", fake_mem.loader(10 * MB, "A"))
    reg.get_or_load("b", fake_mem.loader(10 * MB, "B"))
    reg.get_or_load("c", fake_mem.loader(10 * MB, "C"))  # 30MB > 25MB -> evict "a"

    assert reg.get_or_load("b", _fail).tag == "B"
    assert reg.get_or_load("a", fake_mem.loader(10 * MB, "A2")).tag == "A2"


def test_auto_budget_falls_back_to_unbounded_when_undetectable(monkeypatch):
    monkeypatch.setattr(_cache, "_device_total_bytes", lambda: None)
    reg = ModelRegistry()
    for i in range(5):
        reg.get_or_load(f"k{i}", lambda i=i: _TestModel(f"V{i}"))
    for i in range(5):
        assert reg.get_or_load(f"k{i}", _fail).tag == f"V{i}"


def test_evicts_least_recently_used_over_budget(fake_mem):
    reg = ModelRegistry(budget_mb=25)
    reg.get_or_load("a", fake_mem.loader(10 * MB, "A"))
    reg.get_or_load("b", fake_mem.loader(10 * MB, "B"))
    reg.get_or_load("c", fake_mem.loader(10 * MB, "C"))  # 30MB > 25MB -> evict "a"

    assert reg.get_or_load("b", _fail).tag == "B"
    assert reg.get_or_load("c", _fail).tag == "C"
    # "a" was evicted, so its loader runs again
    assert reg.get_or_load("a", fake_mem.loader(10 * MB, "A2")).tag == "A2"


def test_recent_access_protects_from_eviction(fake_mem):
    reg = ModelRegistry(budget_mb=25)
    reg.get_or_load("a", fake_mem.loader(10 * MB, "A"))
    reg.get_or_load("b", fake_mem.loader(10 * MB, "B"))
    reg.get_or_load("a", _fail)  # touch "a" -> now "b" is least recently used
    reg.get_or_load("c", fake_mem.loader(10 * MB, "C"))  # evicts "b", not "a"

    assert reg.get_or_load("a", _fail).tag == "A"
    assert reg.get_or_load("b", fake_mem.loader(10 * MB, "B2")).tag == "B2"


def test_single_model_over_budget_is_kept(fake_mem):
    reg = ModelRegistry(budget_mb=5)
    assert reg.get_or_load("big", fake_mem.loader(10 * MB, "BIG")).tag == "BIG"
    assert reg.get_or_load("big", _fail).tag == "BIG"  # still resident


def test_clear_is_scoped_by_prefix():
    reg = ModelRegistry(budget_mb=0)
    reg.get_or_load("gliner:x", lambda: _TestModel("X"))
    reg.get_or_load("presidio:y", lambda: _TestModel("Y"))

    reg.clear(prefix="gliner:")

    assert reg.get_or_load("presidio:y", _fail).tag == "Y"  # untouched
    assert reg.get_or_load("gliner:x", lambda: _TestModel("X2")).tag == "X2"  # reloaded


def test_clear_all():
    reg = ModelRegistry(budget_mb=0)
    reg.get_or_load("a", lambda: _TestModel("A"))
    reg.get_or_load("b", lambda: _TestModel("B"))

    reg.clear()

    assert reg.get_or_load("a", lambda: _TestModel("A2")).tag == "A2"
    assert reg.get_or_load("b", lambda: _TestModel("B2")).tag == "B2"


def test_disabled_registry_reloads_every_call():
    calls = {"n": 0}

    def load() -> _TestModel:
        calls["n"] += 1
        return _TestModel("V")

    reg = ModelRegistry(enabled=False)
    assert reg.get_or_load("k", load).tag == "V"
    assert reg.get_or_load("k", load).tag == "V"
    assert calls["n"] == 2  # no caching: loader runs every time
    reg.clear()  # no-op, must not raise
