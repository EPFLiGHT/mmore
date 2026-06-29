"""Shared state for the privacy pipeline graph.

A single ``StateGraph(PrivacyState)`` flows through analyzer -> detector ->
sanitizer (and more later). Each agent contributes a node that reads what
it needs and writes its output back.
"""

from typing import List, Optional

from ..detection.base import PIISpan
from ..policy import PrivacyPolicy
from ..risk import RiskAssessment
from .base import NodeOutput


class PrivacyState(NodeOutput, total=False):
    """Pipeline state and node output for the privacy graph.

    A privacy-specific node output: every agent node returns a partial
    ``PrivacyState``, writing only the fields it produces. ``query`` and
    ``raw_chunks`` are populated by the caller before the graph runs.
    """

    query: str
    raw_chunks: List[str]
    policy: Optional[PrivacyPolicy]
    spans: List[List[PIISpan]]
    risk: Optional[RiskAssessment]
    sanitized_chunks: List[str]
