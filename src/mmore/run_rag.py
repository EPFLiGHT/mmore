import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast
from uuid import UUID

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.documents import Document
from pydantic import BaseModel
from rich.markup import escape

from mmore.privacy.ux import PRIVACY_EMOJI
from mmore.privacy.ux import attach as attach_privacy
from mmore.privacy.ux import detach as detach_privacy
from mmore.profiler import enable_profiling_from_env, profile_function
from mmore.rag.judge import extract_judge_output
from mmore.rag.pipeline import PRIVACY_OUTPUT_KEYS, RAGConfig, RAGPipeline
from mmore.ragcli_helper import RunTimer
from mmore.utils import load_config
from mmore.ux import (
    Color,
    model_loading_seconds,
    paused_seconds,
    progress,
    quiet_noisy_libs,
    setup_logging,
    step_intro,
    step_summary,
)

RAG_NAME = "RAG"
RAG_EMOJI = "🧠"
RAG_PRIVACY_NAME = "RAG WITH PRIVACY"
logger = setup_logging(RAG_NAME, RAG_EMOJI)

load_dotenv()


class BatchGenerationTimer(RunTimer):
    """Accumulates LLM generation time across a RAG batch."""

    def __init__(self, on_generate_start: Optional[Callable[[], None]] = None) -> None:
        super().__init__()
        self.generate_seconds = 0.0
        self._on_generate_start = on_generate_start

    def _start(self, run_id: UUID) -> None:
        self._begin(run_id)
        if self._on_generate_start is not None:
            self._on_generate_start()

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        self._start(run_id)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._start(run_id)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        elapsed = self._elapsed(run_id)
        if elapsed is not None:
            self.generate_seconds += elapsed


@dataclass
class LocalConfig:
    input_file: str
    output_file: str


@dataclass
class APIConfig:
    endpoint: str = "/rag"
    port: int = 8000
    host: str = "0.0.0.0"


@dataclass
class RAGInferenceConfig:
    rag: RAGConfig
    mode: str
    mode_args: Optional[Union[LocalConfig, APIConfig]] = None

    def __post_init__(self):
        if self.mode_args is None and self.mode == "api":
            self.mode_args = APIConfig()


def read_queries(input_file: Union[Path, str]) -> List[Dict[str, str]]:
    with open(input_file, "r") as f:
        return [json.loads(line) for line in f]


# Standard RAG fields (always written to JSON / API)
_RAG_KEYS = ("input", "context", "answer")


def _serialize_document(doc: Union[Document, Dict[str, Any]]) -> Dict[str, Any]:
    """LangChain Documents are not JSON-serializable; clients need plain dicts."""
    if isinstance(doc, dict):
        page_content = doc.get("page_content", doc.get("content", ""))
        metadata = doc.get("metadata", {})
        return {"page_content": page_content, "metadata": dict(metadata)}
    return {"page_content": doc.page_content, "metadata": dict(doc.metadata)}


def _serialize_documents(
    docs: List[Union[Document, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Same as _serialize_document, for the full list written to results.json / API."""
    return [_serialize_document(doc) for doc in docs]


def _to_public_output(pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
    """Bridge pipeline dict (internal keys like docs) to the public RAGOutput / JSON schema."""
    out = {key: pipeline_result[key] for key in _RAG_KEYS if key in pipeline_result}
    out.update(extract_judge_output(pipeline_result))
    # Privacy mode surfaces a PII-free report record + advisory summary, if present
    for key in PRIVACY_OUTPUT_KEYS:
        if key in pipeline_result:
            out[key] = pipeline_result[key]
    docs = pipeline_result.get("docs")
    if docs:
        out["documents"] = _serialize_documents(docs)
    return out


def save_results(results: List[Dict], output_file: Union[Path, str]):
    # JSON-safe dicts (answer, context, judge fields, documents) for JSON export.
    serialized = [_to_public_output(d) for d in results]
    with open(output_file, "w") as f:
        json.dump(serialized, f, indent=2)


class InnerInput(BaseModel):
    input: str
    collection_name: Optional[str] = None


class RAGInput(BaseModel):
    input: InnerInput


class RAGOutput(BaseModel):
    input: Optional[str] = None
    context: Optional[str] = None
    answer: Optional[str] = None
    documents: Optional[List[Dict[str, Any]]] = None
    judge_decision: Optional[str] = None
    judge_reason: Optional[str] = None
    judge_actions: Optional[List[str]] = None
    judge_llm_calls: Optional[int] = None
    judge_steps: Optional[List[Dict[str, Any]]] = None
    hit_max_corrective_steps: Optional[float] = None
    retrieval_metrics: Optional[Dict[str, float]] = None
    retrieval_corrections: Optional[List[Dict[str, Any]]] = None
    privacy_report: Optional[Dict[str, Any]] = None
    sanitized_context: Optional[str] = None


def create_api(rag: RAGPipeline, endpoint: str):
    app = FastAPI(
        title="RAG Pipeline API",
        description="API for question answering using RAG",
        version="2.0",
    )

    # Omit unset judge fields (same rule as _to_public_output for JSON export).
    @app.post(endpoint, response_model=RAGOutput, response_model_exclude_none=True)
    async def run_rag(request: RAGInput):
        # Extract the inner input dict to pass to rag_chain
        pipeline_input = request.input.model_dump()
        output_dict = rag.rag_chain.invoke(pipeline_input)
        return RAGOutput(**_to_public_output(output_dict))

    @app.get("/health")
    def health_check():
        return {"status": "healthy"}

    return app


def _setup_privacy(privacy_config_file: str, mode: str):
    """Load + validate the privacy config and compile its pipeline graph."""
    from mmore.privacy.runner import setup_privacy

    return setup_privacy(privacy_config_file, interactive_ok=mode != "api")


def _privacy_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Privacy counters for the closing card, aggregated over the whole run."""
    from mmore.privacy.schemas.report import PreCloudOutcome

    records = [r["privacy_report"] for r in results if r.get("privacy_report")]
    approved = sum(
        1 for rec in records if rec["gate_outcome"] == PreCloudOutcome.APPROVED
    )
    return {
        "validated & approved": f"{approved}/{max(1, len(results))}",
        "sensitive spans found": sum(rec["detection"]["count"] for rec in records),
        "leakage escalations": sum(rec["adversary_iterations"] for rec in records),
        "human revisions": sum(rec["human_iterations"] for rec in records),
        "verifier warnings": sum(
            warning["count"] for rec in records for warning in rec["verifier_warnings"]
        ),
        "verifier checks failed": sum(
            len(rec["verifier_checks_failed"]) for rec in records
        ),
    }


@profile_function()
def rag(config_file, privacy_config_file: Optional[str] = None):
    """Run RAG in local or API, optionally through the privacy pipeline."""
    quiet_noisy_libs()
    config = load_config(config_file, RAGInferenceConfig)

    privacy_graph, privacy_approver, privacy_config = None, None, None
    if privacy_config_file is not None:
        logger.debug("Privacy mode enabled, building the privacy pipeline...")
        privacy_graph, privacy_approver, privacy_config = _setup_privacy(
            privacy_config_file, config.mode
        )

    if privacy_config is None:
        name, emoji = RAG_NAME, RAG_EMOJI
        about = "Find relevant passages and answer questions about your docs"
        answer_llm = config.rag.llm.llm_name
    else:
        name, emoji = RAG_PRIVACY_NAME, PRIVACY_EMOJI
        about = (
            "Answer questions about your docs with no sensitive data leaving "
            "your machine"
        )
        assert privacy_config.answer is not None
        answer_llm = privacy_config.answer.llm.llm_name

    fields = [
        f"collection: {config.rag.retriever.collection_name}",
        f"LLM: {answer_llm}",
    ]
    if config.mode == "local":
        fields.append(f"answers: {cast(LocalConfig, config.mode_args).output_file}")
    else:
        fields.append(f"mode: {config.mode}")
    step_intro(name, emoji, about, fields)

    logger.debug("Creating the RAG Pipeline...")
    rag_pp = RAGPipeline.from_config(
        config.rag, privacy_graph=privacy_graph, privacy_approver=privacy_approver
    )
    logger.debug("RAG pipeline initialized!")

    if config.mode == "local":
        config_args = cast(LocalConfig, config.mode_args)

        queries = read_queries(config_args.input_file)
        rag_pp.retriever.pop_timings()  # reset accumulators before this run

        # Sequential on purpose: the privacy gate prompts read stdin, so batching
        # the queries would let concurrent prompts interleave.
        results = []
        with progress(total=len(queries), desc="Answering", unit="") as bar:
            stage_labels = {
                "retrieve": "retrieving",
                "rerank": "reranking",
                "generate": "generating",
            }
            rag_pp.retriever.set_stage_callback(
                lambda stage: bar.set_unit(stage_labels.get(stage, stage))
            )
            if privacy_graph is not None:
                attach_privacy(bar)
            timer = BatchGenerationTimer(
                on_generate_start=lambda: bar.set_unit(stage_labels["generate"])
            )

            start = time.time()
            loading_start = model_loading_seconds()
            waiting_start = paused_seconds()
            try:
                for i, query in enumerate(queries, 1):
                    question = str(query.get("input", ""))
                    bar.print_above(
                        f"[{Color.MMORE}]Q{i}/{len(queries)}[/] {escape(question)}"
                    )
                    results.append(
                        rag_pp.rag_chain.invoke(query, config={"callbacks": [timer]})
                    )
                    bar.update(1)
            finally:
                rag_pp.retriever.set_stage_callback(None)
                detach_privacy()
            bar.set_unit("")

        # Model loading and the time you spent at the gate are not the pipeline's
        waiting = paused_seconds() - waiting_start
        elapsed = (
            time.time() - start - (model_loading_seconds() - loading_start) - waiting
        )
        save_results(results, config_args.output_file)

        n = max(1, len(queries))
        retrieve_s, rerank_s = rag_pp.retriever.pop_timings()
        stats: Dict[str, Any] = {
            "queries": len(queries),
            "retrieve": f"{retrieve_s / n:.2f} s/query",
            "rerank": f"{rerank_s / n:.2f} s/query" if rerank_s else "n/a",
        }
        if privacy_graph is None:
            stats["generate"] = f"{timer.generate_seconds / n:.2f} s/query"
        else:
            privacy_s = max(0.0, elapsed - retrieve_s - rerank_s)
            stats["privacy pipeline"] = f"{privacy_s / n:.2f} s/query"
            if waiting:
                stats["waiting on you"] = f"{waiting:.1f} s"
            stats.update(_privacy_stats(results))
        step_summary(name, emoji, elapsed, stats)

    elif config.mode == "api":
        config_args = cast(APIConfig, config.mode_args)

        app = create_api(rag_pp, config_args.endpoint)
        uvicorn.run(app, host=config_args.host, port=config_args.port)

    else:
        raise ValueError(f"Unknown mode: {config.mode}. Should be either api or local")


if __name__ == "__main__":
    enable_profiling_from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", required=True, help="Path to the rag configuration file."
    )
    parser.add_argument(
        "--privacy",
        default=None,
        help="Path to a privacy config; its presence enables privacy mode.",
    )
    args = parser.parse_args()

    rag(args.config_file, privacy_config_file=args.privacy)
