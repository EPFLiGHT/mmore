"""Unit tests for the shared privacy model registry."""

import pytest

from mmore.privacy import _cache
from mmore.privacy._cache import ModelRegistry

MB = 1024 * 1024


class _FakeMem:
    """Stand-in for device memory: loaders bump ``used``, the registry reads it."""

    def __init__(self) -> None:
        self.used = 0

    def __call__(self) -> int:
        return self.used

    def loader(self, size: int, value: str):
        def load() -> str:
            self.used += size
            return value

        return load


def _fail():
    raise AssertionError("loader should not run for a cached key")


@pytest.fixture
def fake_mem(monkeypatch):
    mem = _FakeMem()
    monkeypatch.setattr(_cache, "_device_mem_bytes", mem)
    monkeypatch.setattr(_cache, "_empty_device_cache", lambda: None)
    return mem


def test_loads_once_and_reuses():
    reg = ModelRegistry(budget_mb=0)
    assert reg.get_or_load("k", lambda: "V") == "V"
    assert reg.get_or_load("k", _fail) == "V"


def test_zero_budget_disables_eviction():
    reg = ModelRegistry(budget_mb=0)
    for i in range(5):
        reg.get_or_load(f"k{i}", lambda i=i: f"V{i}")
    for i in range(5):
        assert reg.get_or_load(f"k{i}", _fail) == f"V{i}"


def test_auto_budget_uses_device_total(monkeypatch, fake_mem):
    monkeypatch.setattr(_cache, "_BUDGET_FRACTION", 1.0)
    monkeypatch.setattr(_cache, "_device_total_bytes", lambda: 25 * MB)
    reg = ModelRegistry()  # no budget -> auto-detect
    reg.get_or_load("a", fake_mem.loader(10 * MB, "A"))
    reg.get_or_load("b", fake_mem.loader(10 * MB, "B"))
    reg.get_or_load("c", fake_mem.loader(10 * MB, "C"))  # 30MB > 25MB -> evict "a"

    assert reg.get_or_load("b", _fail) == "B"
    assert reg.get_or_load("a", fake_mem.loader(10 * MB, "A2")) == "A2"


def test_auto_budget_falls_back_to_unbounded_when_undetectable(monkeypatch):
    monkeypatch.setattr(_cache, "_device_total_bytes", lambda: None)
    reg = ModelRegistry()
    for i in range(5):
        reg.get_or_load(f"k{i}", lambda i=i: f"V{i}")
    for i in range(5):
        assert reg.get_or_load(f"k{i}", _fail) == f"V{i}"


def test_evicts_least_recently_used_over_budget(fake_mem):
    reg = ModelRegistry(budget_mb=25)
    reg.get_or_load("a", fake_mem.loader(10 * MB, "A"))
    reg.get_or_load("b", fake_mem.loader(10 * MB, "B"))
    reg.get_or_load("c", fake_mem.loader(10 * MB, "C"))  # 30MB > 25MB -> evict "a"

    assert reg.get_or_load("b", _fail) == "B"
    assert reg.get_or_load("c", _fail) == "C"
    # "a" was evicted, so its loader runs again
    assert reg.get_or_load("a", fake_mem.loader(10 * MB, "A2")) == "A2"


def test_recent_access_protects_from_eviction(fake_mem):
    reg = ModelRegistry(budget_mb=25)
    reg.get_or_load("a", fake_mem.loader(10 * MB, "A"))
    reg.get_or_load("b", fake_mem.loader(10 * MB, "B"))
    reg.get_or_load("a", _fail)  # touch "a" -> now "b" is least recently used
    reg.get_or_load("c", fake_mem.loader(10 * MB, "C"))  # evicts "b", not "a"

    assert reg.get_or_load("a", _fail) == "A"
    assert reg.get_or_load("b", fake_mem.loader(10 * MB, "B2")) == "B2"


def test_single_model_over_budget_is_kept(fake_mem):
    reg = ModelRegistry(budget_mb=5)
    assert reg.get_or_load("big", fake_mem.loader(10 * MB, "BIG")) == "BIG"
    assert reg.get_or_load("big", _fail) == "BIG"  # still resident


def test_clear_is_scoped_by_prefix():
    reg = ModelRegistry(budget_mb=0)
    reg.get_or_load("gliner:x", lambda: "X")
    reg.get_or_load("presidio:y", lambda: "Y")

    reg.clear(prefix="gliner:")

    assert reg.get_or_load("presidio:y", _fail) == "Y"  # untouched
    assert reg.get_or_load("gliner:x", lambda: "X2") == "X2"  # reloaded


def test_clear_all():
    reg = ModelRegistry(budget_mb=0)
    reg.get_or_load("a", lambda: "A")
    reg.get_or_load("b", lambda: "B")

    reg.clear()

    assert reg.get_or_load("a", lambda: "A2") == "A2"
    assert reg.get_or_load("b", lambda: "B2") == "B2"


def test_disabled_registry_reloads_every_call():
    calls = {"n": 0}

    def load():
        calls["n"] += 1
        return "V"

    reg = ModelRegistry(enabled=False)
    assert reg.get_or_load("k", load) == "V"
    assert reg.get_or_load("k", load) == "V"
    assert calls["n"] == 2  # no caching: loader runs every time
    reg.clear()  # no-op, must not raise
