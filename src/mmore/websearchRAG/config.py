# mmore/websearch/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml

from ..rag.llm import LLMConfig  # Reuse the same LLMConfig as RAG


@dataclass
class WebsearchConfig:
    """
    Configuration for WebsearchPipeline.

    Fields:
      rag_config_path:    (str or None) Path to the RAG config YAML. Required if use_rag=True.
      use_rag:            (bool) If True, run RAG first; otherwise skip directly to sub-query generation.
      use_summary:        (bool) If True, run an initial LLM-based summary of the RAG answer.
      input_file:         (str) Path to the JSON file used as “queries” (or RAG output).
      output_file:        (str) Path where the enhanced JSON results will be written.
      input_queries:      (str) Path to queries file.
      n_subqueries:       (int) Number of sub-queries to generate via LLM.
      n_loops:            (int) Number of loops to run the process.
      max_searches:       (int) Max results to fetch per sub-query.
      max_retries:        (int) Max retries for search on rate limit (default: 3).
      search_provider:    (str) Search provider: 'duckduckgo' (default, free), 'tavily' (requires TAVILY_API_KEY, pip install "mmore[rag,websearch]"), or 'keenable' (no key required; optional KEENABLE_API_KEY).
      max_context_tokens: (int) Maximum number of context tokens for constructing prompts (default: 2048).
      fast_tokenizer:     (bool) If True, use a fast heuristic (~4 chars/token) instead of the LLM tokenizer (default: False).
      llm_config:         (dict) Passed to rag.llm.LLMConfig (keys: llm_name, max_new_tokens, temperature, etc.)
      mode:               (str) Mode of operation ("local" or "api").
    """

    rag_config_path: str  # e.g., "../rag/config.yaml"
    output_file: str
    use_rag: bool = False
    use_summary: bool = False
    input_file: Optional[str] = None
    input_queries: Optional[str] = None
    n_subqueries: int = 3
    n_loops: int = 2

    max_searches: int = 10
    max_retries: int = 3
    search_provider: Literal["duckduckgo", "tavily", "keenable"] = "duckduckgo"
    max_context_tokens: int = 2048
    fast_tokenizer: bool = False

    llm_config: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            **{"llm_name": "gpt-4", "max_new_tokens": 1200}
        )
    )
    mode: Literal["local", "api"] = "local"

    def __post_init__(self):
        required_fields = ["n_loops", "n_subqueries", "max_searches", "mode"]
        for field_name in required_fields:
            if not getattr(self, field_name):
                raise ValueError(f"'{field_name}' is a required field.")

    def get_llm_config(self) -> LLMConfig:
        """
        Return the nested llm_config object.
        """
        return self.llm_config

    def access_rag_config(self) -> Dict[str, Any]:
        """
        Access and parse the RAG configuration file defined in `rag_config_path`.

        Returns:
            A dictionary representing the RAG configuration.
        """
        if not self.rag_config_path:
            raise ValueError("The 'rag_config_path' is not defined.")

        # Resolve the full path to the RAG config file
        rag_config_full_path = Path(self.rag_config_path)

        if not rag_config_full_path.exists():
            raise FileNotFoundError(
                f"RAG config file not found at {rag_config_full_path}"
            )

        # Load the RAG configuration
        with open(rag_config_full_path, "r") as file:
            rag_config = yaml.safe_load(file)

        return rag_config
