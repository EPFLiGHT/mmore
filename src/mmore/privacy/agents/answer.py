"""Post-cloud answer model.

Pipeline:  ... -> gate -> [answer] -> verifier -> report
Reads:     sanitized_query, policy (domain prompt), sanitized_chunks
Writes:    answer, answer_backend, answer_model

It receives only the sanitized context that passed the pre-cloud gate
plus the sanitized query and the selected domain prompt.

It must never read the raw chunks or the raw query.
"""

import logging

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import Self

from ...rag.llm import LLMConfig
from ..config import PrivacyConfig, as_privacy_config
from .base import BaseAgent
from .state import PrivacyState

logger = logging.getLogger(__name__)


class AnswerAgent(BaseAgent):
    """The cloud answer model."""

    state_schema = PrivacyState
    node_name = "answer"

    def __init__(
        self,
        config: PrivacyConfig,
        llm_config: LLMConfig,
        system_prompt: str = "",
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self._system_prompt = system_prompt
        super().__init__(config, llm_config=llm_config, checkpointer=checkpointer)

    @classmethod
    def from_config(
        cls,
        config: PrivacyConfig | str | dict,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> Self:
        config = as_privacy_config(config)
        if config.answer is None:
            raise ValueError(
                "Answer model requires 'answer.llm' in the privacy config."
            )
        return cls(
            config,
            config.answer.llm,
            config.answer.system_prompt or "",
            checkpointer=checkpointer,
        )

    @property
    def identity(self) -> tuple[str, str]:
        """Where the answer came from, as (backend, model), for the report."""
        config = self.llm_config
        if config.provider == "HF" and config.base_url is None:
            backend = "local-hf"
        elif config.base_url is not None:
            backend = f"self-hosted ({config.base_url})"
        else:
            backend = config.provider or "unknown"
        return backend, config.llm_name

    def answer(
        self, query: str, sanitized_chunks: list[str], domain_prompt: str = ""
    ) -> str:
        """Answer the query from the sanitized context."""
        context = "\n\n".join(c for c in sanitized_chunks if c).strip()
        system = "\n\n".join(p for p in (domain_prompt, self._system_prompt) if p)
        messages: list[BaseMessage] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        )
        llm = self.llm.bind(**self.llm_config.bind_kwargs)
        return str(llm.invoke(messages).content)

    def _node(self, state: PrivacyState) -> PrivacyState:
        """Graph node: answer from the sanitized context and record the backend."""
        policy = state.get("policy")
        answer = self.answer(
            state.get("sanitized_query", ""),
            list(state.get("sanitized_chunks", [])),
            policy.domain_prompt if policy else "",
        )
        backend, model = self.identity
        return PrivacyState(answer=answer, answer_backend=backend, answer_model=model)
