"""Post-cloud answer model.

Pipeline:  ... -> gate -> [answer] -> verifier -> report
Reads:     query, policy (domain prompt), sanitized_chunks
Writes:    answer, answer_backend, answer_model

It receives nly the sanitized context that passed the pre-cloud gate
plus the query and the selected domain prompt.

It must never reads the raw chunks.
"""

import logging
from typing import List, Optional, Tuple, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from typing_extensions import Self

from ..rag.llm import LLM, LLMConfig
from ..utils import load_config
from .agents.state import PrivacyState
from .config import PrivacyConfig

logger = logging.getLogger(__name__)


class AnswerModel:
    """The cloud answer model."""

    node_name = "answer"

    def __init__(self, llm_config: LLMConfig, system_prompt: Optional[str] = None):
        self._llm_config = llm_config
        self._system_prompt = system_prompt or ""
        self._llm: Optional[BaseChatModel] = None

    @classmethod
    def from_config(cls, config: Union[PrivacyConfig, str, dict]) -> Self:
        if not isinstance(config, PrivacyConfig):
            config = load_config(config, PrivacyConfig)
        if config.answer is None:
            raise ValueError(
                "Answer model requires 'answer.llm' in the privacy config."
            )
        return cls(config.answer.llm, config.answer.system_prompt)

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = LLM.from_config(self._llm_config)
        return self._llm

    @property
    def identity(self) -> Tuple[str, str]:
        cfg = self._llm_config
        if cfg.provider == "HF" and cfg.base_url is None:
            backend = "local-hf"
        elif cfg.base_url is not None:
            backend = f"self-hosted ({cfg.base_url})"
        else:
            backend = cfg.provider or "unknown"
        return backend, cfg.llm_name

    def answer(
        self, query: str, sanitized_chunks: List[str], domain_prompt: str = ""
    ) -> str:
        """Answer the query from the sanitized context."""
        context = "\n\n".join(c for c in sanitized_chunks if c).strip()
        system = "\n\n".join(p for p in [domain_prompt, self._system_prompt] if p)
        messages: List[BaseMessage] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        )
        llm = self.llm.bind(**self._llm_config.bind_kwargs)
        return str(llm.invoke(messages).content)

    def _node(self, state: PrivacyState) -> PrivacyState:
        """Graph node: answer from the sanitized context and record the backend."""
        policy = state.get("policy")
        domain_prompt = policy.domain_prompt if policy else ""
        answer = self.answer(
            state.get("query", ""),
            list(state.get("sanitized_chunks", [])),
            domain_prompt,
        )
        backend, model = self.identity
        return PrivacyState(answer=answer, answer_backend=backend, answer_model=model)
