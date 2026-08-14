"""Unit tests for mmore.rag.judge package (config: examples/rag/config_judge.yaml)."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from mmore.rag.judge import (
    JUDGE_OUTPUT_KEYS,
    JudgeConfig,
    JudgeDecision,
    JudgeResult,
    LLMJudge,
    coerce_decision,
    compute_retrieval_metrics,
    evaluate_metrics,
    extract_judge_output,
    merge_documents,
    parse_json_response,
    record_correction_metrics,
    retrieve_with_judge,
)
from mmore.rag.judge.corrective import apply_corrective_action
from mmore.rag.judge.parsing import extract_llm_text
from mmore.run_rag import RAGInferenceConfig
from mmore.utils import load_config

_inference_cfg = load_config("examples/rag/config_judge.yaml", RAGInferenceConfig)
assert _inference_cfg.rag is not None and _inference_cfg.rag.judge is not None
_CFG: JudgeConfig = _inference_cfg.rag.judge
_THRESH = _CFG.metric_thresholds


def _cfg(**kw):
    return replace(_CFG, **kw)


def _retriever(**kw):
    retriever = MagicMock(**kw)
    retriever.reranker_model = None
    return retriever


def _doc(sim=0.5, id_="1", rerank=None):
    meta = {"rank": 1, "similarity": sim, "id": id_}
    if rerank is not None:
        meta["rerank_score"] = rerank
    return Document(page_content=id_, metadata=meta)


def test_judge_result_proceed():
    r = JudgeResult.proceed("metrics_above_thresholds")
    assert r.decision == JudgeDecision.PROCEED
    assert r.reason == r.exit_reason == "metrics_above_thresholds"
    assert r.llm_invoked is False

    r2 = JudgeResult.proceed(
        "parse_error_fallback",
        llm_invoked=True,
        context_relevance_score=5.0,
    )
    assert r2.llm_invoked and r2.context_relevance_score == 5.0


def test_metrics_thresholds_merge_and_correction_record():
    docs = [_doc(0.9, "1"), _doc(0.5, "2")]
    thresholds = {"min_mean_similarity": 0.35, "min_num_docs": 2}

    assert compute_retrieval_metrics(docs).mean_similarity == pytest.approx(0.7)

    metrics, passed, status = evaluate_metrics(docs, thresholds)
    assert metrics.mean_similarity == pytest.approx(0.7)
    assert passed and "PASS" in status

    metrics2, passed2, _ = evaluate_metrics(
        docs, thresholds, context_relevance_score=3.0
    )
    assert metrics2.context_relevance_score == 3.0 and passed2

    _, fail_rerank, _ = evaluate_metrics(
        [_doc(0.9, "1")], {"min_max_rerank_score": 0.5}
    )
    assert not fail_rerank
    _, pass_rerank, _ = evaluate_metrics(
        [_doc(0.9, "1", rerank=0.8)], {"min_max_rerank_score": 0.5}
    )
    assert pass_rerank

    merged = merge_documents([_doc(0.8, "1")], [_doc(0.8, "1"), _doc(0.6, "2")])
    assert len(merged) == 2 and merged[1].metadata["rank"] == 2

    record = record_correction_metrics(
        "RE_RETRIEVE",
        [_doc(0.2, "1")],
        [_doc(0.8, "2", rerank=2.0)],
        _THRESH,
        context_relevance_score=7.0,
    )
    for section in ("delta", "before", "after"):
        assert "context_relevance_score" not in record.to_dict()[section]
    assert record.delta_dict()["delta_mean_similarity"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    "cfg_kw,raw,expected",
    [
        (
            {
                "allow_re_retrieve": True,
                "allow_add_questions": False,
                "allow_add_context": False,
            },
            "ADD_QUESTIONS",
            JudgeDecision.RE_RETRIEVE,
        ),
        (
            {
                "allow_re_retrieve": False,
                "allow_add_questions": True,
                "allow_add_context": False,
            },
            "RE_RETRIEVE",
            JudgeDecision.ADD_QUESTIONS,
        ),
        (
            {
                "allow_re_retrieve": True,
                "allow_add_questions": False,
                "allow_add_context": False,
            },
            "ADD_CONTEXT",
            JudgeDecision.RE_RETRIEVE,
        ),
        (
            {
                "allow_re_retrieve": False,
                "allow_add_questions": False,
                "allow_add_context": True,
            },
            "ADD_QUESTIONS",
            JudgeDecision.PROCEED,
        ),
        (
            {
                "allow_re_retrieve": False,
                "allow_add_questions": False,
                "allow_add_context": False,
            },
            "RE_RETRIEVE",
            JudgeDecision.PROCEED,
        ),
        (
            {
                "allow_re_retrieve": True,
                "allow_add_questions": True,
                "allow_add_context": False,
            },
            "ADD_QUESTIONS",
            JudgeDecision.ADD_QUESTIONS,
        ),
    ],
)
def test_coerce_decision_fallback(cfg_kw, raw, expected):
    decision, _, raw_decision = coerce_decision(raw, _cfg(**cfg_kw))
    assert decision == expected and raw_decision == raw


def test_parse_json_from_full_llama_response():
    raw = (
        "Retrieval metrics: {'num_docs': 5.0}\n"
        "Allowed actions: ['PROCEED', 'ADD_QUESTIONS']\n"
        "<|eot_id|>assistant\n\n"
        '{"decision":"ADD_QUESTIONS","extra_questions":["q1","q2"],"reason":"split"}'
    )
    parsed = parse_json_response(raw)
    assert parsed["decision"] == "ADD_QUESTIONS"
    assert parsed["extra_questions"] == ["q1", "q2"]


def test_extract_llm_text_strips_llama_chat_template():
    raw = (
        "<|begin_of_text|>systemYou are a judge.<|eot_id|>"
        "userQuestion?<|eot_id|>assistant"
        '{"decision":"ADD_QUESTIONS","extra_questions":["q1"]}'
        "<|eot_id|>"
    )
    assert (
        extract_llm_text(raw) == '{"decision":"ADD_QUESTIONS","extra_questions":["q1"]}'
    )


def test_step_record_includes_llm_response():
    from mmore.rag.judge.decisions import step_record

    result = JudgeResult(
        decision=JudgeDecision.ADD_QUESTIONS,
        extra_questions=["q1"],
        llm_invoked=True,
        raw_llm_response='{"decision":"ADD_QUESTIONS","extra_questions":["q1"]}',
        exit_reason="llm_corrective",
    )
    record = step_record(0, result, "main query", 5)
    assert record["llm_response"] == result.raw_llm_response
    assert record["extra_questions"] == ["q1"]


def test_parse_json_repairs_trailing_comma_and_python_literals():
    p = parse_json_response('{"decision":"RE_RETRIEVE","context_relevance_score":4,}')
    assert p["decision"] == "RE_RETRIEVE" and p["context_relevance_score"] == 4
    p = parse_json_response('{"decision":"PROCEED","sufficient":True}')
    assert p["sufficient"] is True


def test_parse_json_nested_retrieve_params_with_prefix():
    raw = (
        'Summary: {"k": 10}\n'
        '{"decision":"RE_RETRIEVE","retrieve_params":{"k":10},"reason":"weak"}'
    )
    parsed = parse_json_response(raw)
    assert parsed["decision"] == "RE_RETRIEVE"
    assert parsed["retrieve_params"] == {"k": 10}


def test_parse_json_valid_prefix_and_invalid_decision_block():
    raw = 'Summary: {"k": 10}\n{"decision":"RE_RETRIEVE","context_relevance_score":4,}'
    parsed = parse_json_response(raw)
    assert parsed["decision"] == "RE_RETRIEVE"
    assert parsed["context_relevance_score"] == 4


@pytest.mark.parametrize(
    "cfg_kw,docs,llm_json,invoke,decision,reason_sub,extra",
    [
        (
            {"force_corrective_action": "PROCEED", "metric_thresholds": _THRESH},
            [_doc(0.2)],
            None,
            False,
            "PROCEED",
            "force_corrective",
            {},
        ),
        (
            {"metric_thresholds": {"min_mean_similarity": 0.3, "min_num_docs": 1}},
            [_doc(0.9)],
            None,
            False,
            "PROCEED",
            "metrics_above",
            {"exit_reason": "metrics_above_thresholds"},
        ),
        (
            {},
            [_doc(0.2)],
            '{"decision":"RE_RETRIEVE","reason":"weak"}',
            True,
            "RE_RETRIEVE",
            "weak",
            {"exit_reason": "llm_corrective"},
        ),
        (
            {"metric_thresholds": _THRESH},
            [_doc(0.2)],
            '{"decision":"PROCEED","reason":"ok","context_relevance_score":8}',
            True,
            "PROCEED",
            "ok",
            {"context_relevance_score": 8.0, "exit_reason": "llm_proceed"},
        ),
        (
            {"metric_thresholds": {"min_num_docs": 1}},
            [],
            "not json",
            True,
            "PROCEED",
            "parse_error",
            {},
        ),
    ],
)
def test_llm_judge_evaluate(
    cfg_kw, docs, llm_json, invoke, decision, reason_sub, extra
):
    llm = MagicMock()
    if llm_json:
        llm.invoke.return_value = MagicMock(content=llm_json)
    r = LLMJudge(llm, _cfg(**cfg_kw)).evaluate("q?", docs)
    assert llm.invoke.called == invoke
    assert r.decision == JudgeDecision(decision)
    assert reason_sub in r.reason
    for key, value in extra.items():
        assert getattr(r, key) == value


def test_llm_judge_result_from_parsed():
    parsed = {
        "decision": "RE_RETRIEVE",
        "reason": "weak retrieval",
        "context_relevance_score": 4,
        "retrieve_params": {"k": 10},
    }
    result = LLMJudge(MagicMock(), _cfg())._result_from_parsed(parsed)
    assert result.decision == JudgeDecision.RE_RETRIEVE
    assert result.reason == "weak retrieval"
    assert result.exit_reason == "llm_corrective"
    assert result.context_relevance_score == 4.0
    assert result.retrieve_params == {"k": 10}


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            {
                "name": "proceed",
                "max_steps": 1,
                "invoke_docs": [[_doc(0.9)]],
                "evaluate_results": [
                    JudgeResult(decision=JudgeDecision.PROCEED),
                ],
                "checks": lambda out, judge, retriever: (
                    all(
                        k in out
                        for k in JUDGE_OUTPUT_KEYS
                        if k != "retrieval_corrections"
                    )
                    and extract_judge_output(out)["judge_decision"] == "PROCEED"
                    and "retrieval_corrections" not in extract_judge_output(out)
                    and out["judge_actions"] == []
                    and judge.evaluate.call_count == 1
                ),
            },
            id="proceed",
        ),
        pytest.param(
            {
                "name": "re_retrieve_success",
                "max_steps": 2,
                "retriever_k": 5,
                "invoke_docs": [[_doc(0.2)], [_doc(0.8, "2", rerank=2.0)]],
                "evaluate_results": [
                    JudgeResult(
                        decision=JudgeDecision.RE_RETRIEVE,
                        exit_reason="llm_corrective",
                        llm_invoked=True,
                        retrieve_params={"k": 10},
                    ),
                    JudgeResult(
                        decision=JudgeDecision.PROCEED,
                        exit_reason="metrics_above_thresholds",
                    ),
                ],
                "checks": lambda out, judge, retriever: (
                    out["judge_actions"] == ["RE_RETRIEVE"]
                    and out["retrieval_corrections"][0]["action"] == "RE_RETRIEVE"
                    and out["judge_llm_calls"] == 1
                    and judge.evaluate.call_count == 2
                ),
            },
            id="re_retrieve",
        ),
        pytest.param(
            {
                "name": "max_corrective_steps",
                "max_steps": 1,
                "retriever_k": 5,
                "invoke_docs": [[_doc(0.2)], [_doc(0.8, "2", rerank=2.0)]],
                "evaluate_results": [
                    JudgeResult(
                        decision=JudgeDecision.RE_RETRIEVE,
                        exit_reason="llm_corrective",
                        llm_invoked=True,
                        retrieve_params={"k": 10},
                    ),
                    JudgeResult(
                        decision=JudgeDecision.RE_RETRIEVE,
                        exit_reason="llm_corrective",
                        llm_invoked=True,
                    ),
                ],
                "checks": lambda out, judge, retriever: (
                    out["judge_reason"] == "max_corrective_steps"
                    and out["hit_max_corrective_steps"] == 1.0
                    and out["judge_llm_calls"] == 2
                    and out["judge_actions"] == ["RE_RETRIEVE"]
                ),
            },
            id="max_steps",
        ),
        pytest.param(
            {
                "name": "metrics_stop_after_correction",
                "max_steps": 2,
                "metric_thresholds": _THRESH,
                "invoke_docs": [
                    [_doc(0.2)],
                    [
                        _doc(0.9, "2", rerank=2.0),
                        _doc(0.85, "3", rerank=2.0),
                        _doc(0.88, "4", rerank=2.0),
                    ],
                ],
                "evaluate_results": [
                    JudgeResult(
                        decision=JudgeDecision.RE_RETRIEVE, retrieve_params={"k": 10}
                    ),
                    JudgeResult(
                        decision=JudgeDecision.PROCEED,
                        reason="metrics_above_thresholds",
                    ),
                ],
                "checks": lambda out, judge, retriever: (
                    out["judge_decision"] == "PROCEED"
                    and out["retrieval_metrics"]["thresholds_met"] == 1.0
                    and judge.evaluate.call_count == 2
                ),
            },
            id="metrics_stop",
        ),
    ],
)
def test_retrieve_with_judge(scenario):
    retriever, judge = _retriever(), MagicMock()
    if "retriever_k" in scenario:
        retriever.k = scenario["retriever_k"]
    retriever.invoke.side_effect = scenario["invoke_docs"]

    cfg_kw = {"max_corrective_steps": scenario["max_steps"]}
    if "metric_thresholds" in scenario:
        cfg_kw["metric_thresholds"] = scenario["metric_thresholds"]
    judge.config = _cfg(**cfg_kw)
    judge.evaluate.side_effect = scenario["evaluate_results"]

    out = retrieve_with_judge(retriever, judge, {"input": "q?"})
    assert scenario["checks"](out, judge, retriever)


def test_rerank_after_merge_on_add_context():
    retriever = _retriever()
    retriever.reranker_model = MagicMock()
    retriever._get_web_documents.return_value = [_doc(0.5, "web")]
    reranked = [_doc(0.9, "onprem", rerank=2.0), _doc(0.5, "web", rerank=1.0)]
    retriever.rerank.return_value = reranked

    docs = apply_corrective_action(
        retriever,
        _cfg(rerank_after_merge=True),
        {"input": "q?"},
        [_doc(0.9, "onprem", rerank=2.0)],
        JudgeResult(decision=JudgeDecision.ADD_CONTEXT, web_query="web q"),
    )
    retriever.rerank.assert_called_once_with("q?", retriever.rerank.call_args[0][1])
    assert docs == reranked
