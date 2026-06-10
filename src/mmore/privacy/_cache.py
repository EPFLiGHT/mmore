"""Process-wide cache for expensive privacy models, with optional LRU eviction.

A single shared ``MODEL_REGISTRY`` is used by every privacy engine, so the whole
pipeline obeys one caching policy. Models load lazily once per key and are
reused across agents with least-recently-used eviction policy. The budget defaults
to a fraction of the active device's memory, change it with env
``MMORE_PRIVACY_MODEL_BUDGET_MB`` (set ``0`` to disable eviction).

Set env ``MMORE_PRIVACY_MODEL_CACHE=0`` to disable the registry entirely.
"""

import gc
import logging
import os
import threading
from collections import OrderedDict
from typing import Callable, Optional, Tuple, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")

_BUDGET_ENV = "MMORE_PRIVACY_MODEL_BUDGET_MB"
_CACHE_ENABLED_ENV = "MMORE_PRIVACY_MODEL_CACHE"
_MB = 1024 * 1024
_BUDGET_FRACTION = 0.9


def _device_mem_bytes() -> int:
    """Bytes currently allocated on the active device."""
    import torch

    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory()
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    import psutil

    return psutil.Process().memory_info().rss


def _empty_device_cache() -> None:
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _device_total_bytes() -> Optional[int]:
    """Total memory of the active device."""
    import torch

    if torch.backends.mps.is_available():
        return torch.mps.recommended_max_memory()
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory
    try:
        import psutil
    except ImportError:
        return None
    return psutil.virtual_memory().total


class ModelRegistry:
    """Thread-safe LRU cache of loaded models shared across privacy engines."""

    def __init__(self, budget_mb: Optional[float] = None, enabled: bool = True) -> None:
        self._enabled = enabled
        self._budget_mb = budget_mb
        self._budget_bytes: Optional[int] = None
        self._budget_ready = False
        self._entries: "OrderedDict[str, Tuple[object, int]]" = OrderedDict()
        self._total = 0
        self._lock = threading.Lock()

    def _budget(self) -> Optional[int]:
        """Resolve the byte budget once, auto-detecting device memory if unset."""
        if not self._budget_ready:
            if self._budget_mb is None:
                total = _device_total_bytes()
                self._budget_bytes = int(total * _BUDGET_FRACTION) if total else None
            elif self._budget_mb > 0:
                self._budget_bytes = int(self._budget_mb * _MB)
            self._budget_ready = True
        return self._budget_bytes

    def get_or_load(self, key: str, loader: Callable[[], T]) -> T:
        """Return the model for ``key``, loading it via ``loader`` at most once."""
        if not self._enabled:
            return loader()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return cast(T, entry[0])

            budget = self._budget()
            if budget is None:
                value = loader()
                self._entries[key] = (value, 0)
                return value

            before = _device_mem_bytes()
            value = loader()
            size = max(_device_mem_bytes() - before, 0)
            self._entries[key] = (value, size)
            self._total += size
            self._evict_until_within_budget(budget, protect=key)
            return value

    def _evict_until_within_budget(self, budget: int, protect: str) -> None:
        # Iterating on the dict yields keys least-recently-used first
        for key in list(self._entries):
            if self._total <= budget:
                return
            if key == protect:
                continue
            _, size = self._entries.pop(key)
            self._total -= size
            gc.collect()
            _empty_device_cache()
            logger.info("Evicted cached model %r (%d MB)", key, size // _MB)
        if self._total > budget:
            logger.warning(
                "Model %r alone exceeds the model budget; keeping it resident", protect
            )

    def clear(self, prefix: Optional[str] = None) -> None:
        """Drop cached models whose key starts with ``prefix`` (all if ``None``)."""
        if not self._enabled:
            return
        with self._lock:
            freed = 0
            for key in [
                k for k in self._entries if prefix is None or k.startswith(prefix)
            ]:
                _, size = self._entries.pop(key)
                self._total -= size
                freed += size
            if freed > 0:
                gc.collect()
                _empty_device_cache()


def _default_budget_mb() -> Optional[float]:
    raw = os.environ.get(_BUDGET_ENV)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, ignoring", _BUDGET_ENV, raw)
        return None


def _cache_enabled() -> bool:
    raw = os.environ.get(_CACHE_ENABLED_ENV)
    return raw is None or raw.strip().lower() not in {"0", "false", "off", "no"}


MODEL_REGISTRY = ModelRegistry(budget_mb=_default_budget_mb(), enabled=_cache_enabled())
