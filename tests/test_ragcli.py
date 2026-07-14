from unittest.mock import MagicMock

from pymilvus.exceptions import MilvusException

from mmore.run_ragcli import RagCLI, TimingHandler


def _make_cli(answer: str) -> RagCLI:
    cli = RagCLI("unused.yaml")
    cli.ragConfig = MagicMock()
    cli.ragConfig.rag.retriever.use_web = False
    cli.ragConfig.rag.llm.llm_name = "test-model"
    cli.ragConfig.rag.llm.provider = "HF"
    ragpp = MagicMock(return_value=[{"input": "q", "answer": answer, "docs": []}])
    ragpp.llm.tokenizer.encode = lambda text: list(range(5))
    cli.ragPP = ragpp
    return cli


def test_do_rag_and_print_answer_output_full_answer(capsys):
    """The CLI must print the whole answer string, not its last character."""
    answer = "Barack Obama was born on August 4, 1961."
    cli = _make_cli(answer)

    results, timings = cli.do_rag("When was Barack Obama born?")
    cli.print_answer(results, timings)

    assert answer in capsys.readouterr().out


def test_cli_ception_handles_milvus_failure(capsys, monkeypatch):
    """A locked/unreachable local DB must not crash the CLI."""
    cli = _make_cli("unused")
    cli.ragPP = None
    monkeypatch.setattr(
        cli,
        "initialize_ragpp",
        MagicMock(side_effect=MilvusException(message="Open local milvus failed")),
    )

    cli.cli_ception()

    out = capsys.readouterr().out
    assert "Failed to open the document database" in out
    assert "pkill -f milvus_lite/lib/milvus" in out


def test_print_answer_shows_metrics(capsys):
    cli = _make_cli("Some answer.")
    timings = TimingHandler()
    timings.retrieval_time = 0.42
    timings.generation_time = 11.3
    timings.completion_tokens = 142

    results = [
        {
            "input": "q",
            "answer": "Some answer.",
            "context": "some context",
            "docs": [
                {"metadata": {"similarity": 0.47}},
                {"metadata": {"similarity": 0.40}},
            ],
        }
    ]
    cli.print_answer(results, timings)

    out = capsys.readouterr().out
    assert "test-model (local)" in out
    assert "model test-model" not in out
    assert "retrieval 0.42s" in out
    assert "generation 11.30s" in out
    assert "2 chunks" in out
    assert "5 context tokens" in out
    assert "142 tokens @ 13 tok/s" in out
    assert "top score 0.47" in out
