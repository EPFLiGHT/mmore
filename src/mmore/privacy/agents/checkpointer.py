"""Checkpoint builder using LangGraph.

``MemorySaver`` is intended for tests and temporary runs, and
``SqliteSaver`` for persistence across processes.
"""

import sqlite3
from enum import Enum
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from .config import AgentConfig


class Checkpointer(Enum):
    MEMORY = "memory"
    SQLITE = "sqlite"


def build_checkpointer(config: AgentConfig) -> BaseCheckpointSaver | None:
    """Build the checkpointer named by ``config``, or ``None`` when unset."""
    if config.checkpointer is None:
        return None
    match Checkpointer(config.checkpointer):
        case Checkpointer.MEMORY:
            return MemorySaver()
        case Checkpointer.SQLITE:
            if not config.checkpoint_path:
                raise ValueError(
                    "'sqlite' checkpointer requires `checkpoint_path` to be set"
                )
            path = Path(config.checkpoint_path)
            path.parent.mkdir(exist_ok=True, parents=True)
            return SqliteSaver(sqlite3.connect(path, check_same_thread=False))
