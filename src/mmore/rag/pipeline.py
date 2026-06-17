"""
Example implementation:
RAG pipeline.
Integrates Milvus retrieval with HuggingFace text generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)

from ..utils import load_config
from .judge import JUDGE_OUTPUT_KEYS, JudgeConfig, LLMJudge, retrieve_with_judge
from .judge.llm import judge_llm_from_config
from .llm import LLM, LLMConfig
from .retriever import Retriever, RetrieverConfig
from .types import MMOREInput, MMOREOutput

DEFAULT_PROMPT = """\
Use the following context to answer the questions. If none of the context answer the question, just say you don't know.

Context:
{context}
"""

# Privacy fields surfaced alongside the answer when privacy mode is on. Both are
# PII-free: a structured report record and an advisory type+count summary.
PRIVACY_OUTPUT_KEYS = ("privacy_report", "privacy_warnings")


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""

    retriever: RetrieverConfig
    llm: LLMConfig = field(default_factory=lambda: LLMConfig(llm_name="gpt2"))
    system_prompt: str = DEFAULT_PROMPT
    judge: Optional[JudgeConfig] = None


class RAGPipeline:
    """Main RAG pipeline combining retrieval and generation."""

    retriever: Retriever
    llm: BaseChatModel
    prompt_template: Union[str, ChatPromptTemplate]

    def __init__(
        self,
        retriever: Retriever,
        prompt_template: Union[str, ChatPromptTemplate],
        llm: BaseChatModel,
        judge: Optional[LLMJudge] = None,
        privacy_graph: Optional[Any] = None,
    ):
        # Get modules
        self.retriever = retriever
        self.prompt = prompt_template
        self.llm = llm
        self.judge = judge
        # Compiled privacy pipeline; when set, it replaces the answer step.
        self.privacy_graph = privacy_graph

        # Build the rag chain
        self.rag_chain = RAGPipeline._build_chain(
            self.retriever,
            RAGPipeline.format_docs,
            self.prompt,
            self.llm,
            self.judge,
            self.privacy_graph,
        )

    def __str__(self):
        return str(self.rag_chain)

    @classmethod
    def from_config(cls, config: str | RAGConfig, privacy_graph: Optional[Any] = None):
        if isinstance(config, str):
            config = load_config(config, RAGConfig)

        retriever = Retriever.from_config(config.retriever)
        llm = LLM.from_config(config.llm)
        judge = (
            LLMJudge(llm=judge_llm_from_config(config.judge.llm), config=config.judge)
            if config.judge
            else None
        )
        chat_template = ChatPromptTemplate.from_messages(
            [("system", config.system_prompt), ("human", "{input}")]
        )

        return cls(retriever, chat_template, llm, judge, privacy_graph)

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Format documents for prompt."""
        return "\n\n".join(
            f"[{doc.metadata['rank']}] {doc.page_content}" for doc in docs
        )

    @staticmethod
    def _build_chain(
        retriever, format_docs, prompt, llm, judge=None, privacy_graph=None
    ) -> Runnable:
        validate_input = RunnableLambda(
            lambda x: MMOREInput.model_validate(x).model_dump()
        )

        def make_output(x):
            """Validate the output of the LLM and keep only the actual answer of the assistant"""
            res_dict = MMOREOutput.model_validate(x).model_dump()
            res_dict["answer"] = res_dict["answer"].split("<|im_start|>assistant\n")[-1]
            # Expose formatted context and judge correction logs in the API response (context is not on MMOREOutput).
            for key in ("context", *JUDGE_OUTPUT_KEYS, *PRIVACY_OUTPUT_KEYS):
                if key in x:
                    res_dict[key] = x[key]

            return res_dict

        validate_output = RunnableLambda(make_output)

        rag_chain_from_docs = prompt | llm | StrOutputParser()

        # Only retrieval differs (retriever vs judge); format context and generate answer unchanged.
        if judge is not None:
            # retrieve with judge
            def retrieval_with_judge(state: Dict[str, Any]) -> Dict[str, Any]:
                return retrieve_with_judge(retriever, judge, state)

            retrieval_step: Runnable = RunnableLambda(retrieval_with_judge)
        else:
            # retrieve without judge
            retrieval_step = RunnablePassthrough.assign(docs=retriever)

        with_context = retrieval_step.assign(context=lambda x: format_docs(x["docs"]))
        if privacy_graph is not None:
            # Privacy mode swaps only the answer step: the verified answer comes
            # from the privacy graph driven over the retrieved chunks.
            answer_step: Runnable = RunnableLambda(
                RAGPipeline._privacy_answer_step(privacy_graph)
            )
            core_chain = with_context | answer_step
        else:
            core_chain = with_context.assign(answer=rag_chain_from_docs)

        return validate_input | core_chain | validate_output

    @staticmethod
    def _privacy_answer_step(privacy_graph):
        """Answer step that routes the retrieved chunks through the privacy graph.

        Returns the chain state updated with the verified ``answer`` and the
        PII-free report fields, mirroring the ``.assign(answer=...)`` it replaces.
        """
        from dataclasses import asdict

        from ..privacy.runner import run_privacy_query

        def step(x: Dict[str, Any]) -> Dict[str, Any]:
            docs: List[Document] = x["docs"]
            raw_chunks = [doc.page_content for doc in docs]
            result = run_privacy_query(privacy_graph, x["input"], raw_chunks)
            updated = dict(x)
            updated["answer"] = result.answer
            if result.record is not None:
                updated["privacy_report"] = asdict(result.record)
                updated["privacy_warnings"] = [
                    {"kind": w.kind.value, "count": w.count}
                    for w in result.record.advisory_warnings
                ]
            return updated

        return step

    def __call__(
        self,
        queries: Dict[str, Any] | List[Dict[str, Any]],
        return_dict: bool = False,
        config: Optional[RunnableConfig] = None,
    ) -> List[Dict[str, Any]]:
        if isinstance(queries, dict):
            queries_list = [queries]
        else:
            queries_list = queries

        results = self.rag_chain.batch(queries_list, config=config)

        if return_dict:
            return results
        else:
            return [result["answer"] for result in results]
