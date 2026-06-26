"""Presentation helpers for the RAG CLI: timing/token metrics collection."""

import time
from threading import Lock
from typing import Dict, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

# ----------------------------- Timing metrics ------------------------------ #


class RunTimer(BaseCallbackHandler):
    """Thread-safe handler that stamps perf_counter at run start and diffs at end."""

    def __init__(self) -> None:
        self._starts: Dict[UUID, float] = {}
        self._lock = Lock()

    def _begin(self, run_id: UUID) -> None:
        with self._lock:
            self._starts[run_id] = time.perf_counter()

    def _elapsed(self, run_id: UUID) -> Optional[float]:
        with self._lock:
            start = self._starts.pop(run_id, None)
        return None if start is None else time.perf_counter() - start


class TimingHandler(RunTimer):
    """Collects retrieval/generation wall times and token usage from callbacks."""

    def __init__(self) -> None:
        super().__init__()
        self.retrieval_time: Optional[float] = None
        self.generation_time: Optional[float] = None
        self.completion_tokens: Optional[int] = None

    def on_retriever_start(self, serialized, query, *, run_id, **kwargs) -> None:
        self._begin(run_id)

    def on_retriever_end(self, documents, *, run_id, **kwargs) -> None:
        elapsed = self._elapsed(run_id)
        if elapsed is not None:
            self.retrieval_time = elapsed

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        self._begin(run_id)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._begin(run_id)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        elapsed = self._elapsed(run_id)
        if elapsed is not None:
            self.generation_time = elapsed
        self.completion_tokens = _output_tokens(response)


def _output_tokens(response) -> int | None:
    """Generated-token count if the provider reported it (specific to API models)."""
    try:
        usage = response.generations[0][0].message.usage_metadata
        if usage and usage.get("output_tokens"):
            return usage["output_tokens"]
    except (AttributeError, IndexError, TypeError):
        pass
    usage = (response.llm_output or {}).get("token_usage", {})
    return usage.get("completion_tokens") or usage.get("output_tokens")
