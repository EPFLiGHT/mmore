import argparse
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from huggingface_hub import model_info
from huggingface_hub.errors import HfHubHTTPError
from pymilvus.exceptions import MilvusException

from mmore.privacy.config import PrivacyConfig
from mmore.privacy.ux import ANSWER_STAGE, PRIVACY_EMOJI, tool_name
from mmore.privacy.verification import WarningKind
from mmore.profiler import enable_profiling_from_env, profile_function
from mmore.rag.pipeline import PRIVACY_CHUNKS_KEY, RAGPipeline
from mmore.ragcli_helper import TimingHandler
from mmore.run_rag import RAGInferenceConfig
from mmore.utils import load_config
from mmore.ux import (
    Color,
    Spinner,
    live_status,
    paused_seconds,
    plural,
    print_in_color,
    quiet_noisy_libs,
    setup_logging,
    str_brand,
    str_in_color,
)

RAG_NAME = "RAG"
RAG_EMOJI = "🧠"
logger = setup_logging(RAG_NAME, RAG_EMOJI)

_CONTEXT_CMD = "/context"

_VERIFIER_FINDINGS = {
    WarningKind.RESIDUAL_LEAKAGE: "the answer may still reveal sensitive data",
    WarningKind.FAITHFULNESS: "the answer may not follow from the context",
}


def _join(names: List[str]) -> str:
    """`a`, `a and b`, or `a, b and c`."""
    if len(names) < 2:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _print_stats(parts: List[str]) -> None:
    """One grey line under the answer: `a | b | c`."""
    print(str_in_color(" | ".join(parts), Color.GRAY))


def _privacy_seconds(report: Dict[str, Any]) -> float:
    """What the pipeline itself cost: every agent but the answer model."""
    stages = report.get("stage_seconds") or {}
    return sum(seconds for name, seconds in stages.items() if name != ANSWER_STAGE)


class RagCLI:
    def __init__(self, config_file: str, privacy_config_file: Optional[str] = None):
        self.ragConfig: Optional[RAGInferenceConfig] = None
        self.ragPP = None
        self.modified: bool = (
            False  # flag to indicate if the configuration has been modified
        )
        self.config_file = config_file
        self.privacy_config_file = privacy_config_file
        self.privacy_on = privacy_config_file is not None
        self.privacy_config: Optional[PrivacyConfig] = None
        self.privacy_graph: Optional[Any] = None
        self.privacy_approver: Optional[Callable[[dict], object]] = None
        self.last_result: Optional[Dict[str, Any]] = None

    def launch_cli(self):
        quiet_noisy_libs(hide_info=True)
        print_in_color(
            "Welcome to this RAG command-line interface! 🧠", Color.MMORE, bold=True
        )
        print(
            f"\nPress {str_brand('Enter', bold=True)} to start asking questions about your documents.\n"
        )
        print(
            f"Other commands:\n\
        {str_brand('config')} : see the current config \n\
        {str_brand('setK')} : set the number of documents to retrieve \n\
        {str_brand('setModel')} : set the model for generation \n\
        {str_brand('setWebrag')} : decide whether to use web rag \n\
        {str_brand('privacy')} : sanitize the context before the answer model sees it (privacy on/off) \n\
        {str_brand('adversary')} : probe the sanitized context for leaks (adversary on/off) \n\
        {str_brand('hitl')} : approve the sanitized context yourself before it is sent (hitl on/off) \n\
        {str_brand('help')} : learn more about a command (help <command>) \n\
        {str_brand('exit')} : exit the CLI"
        )
        while True:
            try:
                cmd = input("> ").strip()
                if cmd == "exit":
                    print("Goodbye!")
                    break
                elif cmd == "help":
                    print(
                        f"Press {str_brand('Enter')} (or type rag) to start asking questions about your documents.\nOther commands are: config, setK, setModel, setWebrag, exit. To learn more about usage of a specific command, use the following: \n help <command>"
                    )
                elif cmd.startswith("help "):
                    command = cmd.split(" ", 1)[1]
                    if command == "help":
                        print(
                            "To see a list of commands, use the command 'help'. To learn more about usage of a specific command, use the following: \n help <command>"
                        )
                    elif command == "config":
                        print("Print the current configuration.")
                    elif command == "rag":
                        print(
                            "Start a chat session to ask questions about your documents. Type /bye to exit."
                        )
                    elif command == "setK":
                        print(
                            "Use the command in the following way: 'setK <k>', for a positive integer k. This will set the number of documents to retrieve during RAG."
                        )
                    elif command == "setModel":
                        print(
                            "Use the command in the following way: 'setModel <model_path>', where model_path is the huggingface path to the model you'd like to use."
                        )
                    elif command == "setWebrag":
                        print(
                            "Use the command in the following way: 'setWebrag <bool>', where bool is either True or False. This will determine if a web search is done during RAG."
                        )
                    elif command == "privacy":
                        print(
                            "Sanitize the retrieved context before the answer model sees it.\n"
                            "Usage: 'privacy on', 'privacy off', or 'privacy <path to a privacy YAML>' to load a config."
                        )
                    elif command == "adversary":
                        print(
                            "Attack the sanitized context: if anything is recovered, escalate the policy and sanitize again.\n"
                            "Usage: 'adversary on' or 'adversary off'. Only used while privacy mode is on."
                        )
                    elif command == "hitl":
                        print(
                            "Stop before the cloud call to approve, revise, or abort the sanitized context.\n"
                            f"Off answers without asking; {_CONTEXT_CMD} shows the context after the fact.\n"
                            "Usage: 'hitl on' or 'hitl off'. Only used while privacy mode is on."
                        )
                    elif command == "exit":
                        print("Exit the CLI.")
                    else:
                        print("Sorry, this command does not exist.")

                elif cmd == "config":
                    self.init_config()
                    assert self.ragConfig is not None

                    confrag = self.ragConfig.rag
                    print(
                        f"k: {str_in_color(confrag.retriever.k, Color.BLUE)} \nmodel: {str_in_color(confrag.llm.llm_name, Color.BLUE)} \nuse web for rag: {str_in_color(confrag.retriever.use_web, Color.BLUE)} \nprivacy: {str_in_color('on' if self.privacy_on else 'off', Color.BLUE)}"
                    )
                    self.print_privacy_config()
                elif cmd.startswith("privacy"):
                    self.set_privacy(cmd)
                elif cmd.startswith("adversary"):
                    self.set_adversary(cmd)
                elif cmd.startswith("hitl"):
                    self.set_hitl(cmd)
                elif cmd.startswith("greet "):
                    name = cmd.split(" ", 1)[1]
                    print(f"Hello, {str_in_color(name, Color.YELLOW, True)}!")
                elif cmd.startswith("setK "):
                    k_str = cmd.split(" ", 1)[1]
                    try:
                        k = int(k_str)
                        if k > 0:
                            print(k)
                            self.init_config()
                            assert self.ragConfig is not None

                            self.ragConfig.rag.retriever.k = k
                            self.modified = True
                        else:
                            print("Please enter a positive integer.")
                    except ValueError:
                        print("Invalid input. Please enter a valid integer.")
                elif cmd.startswith("setModel "):
                    new_model = cmd.split(" ", 1)[1]
                    print(new_model)
                    valid, message = is_valid_model_path(new_model)
                    if valid:
                        print(message)
                        self.init_config()
                        assert self.ragConfig is not None

                        self.ragConfig.rag.llm.llm_name = new_model
                        self.modified = True
                    else:
                        print(message)

                elif cmd.startswith("setWebrag "):
                    res = cmd.split(" ", 1)[1].lower()
                    if res in ["true", "false"]:
                        self.init_config()
                        assert self.ragConfig is not None

                        old = self.ragConfig.rag.retriever.use_web
                        self.ragConfig.rag.retriever.use_web = (
                            True if res == "true" else False
                        )
                        self.modified = (
                            False
                            if old == self.ragConfig.rag.retriever.use_web
                            else True
                        )
                    else:
                        print(
                            f"Invalid output. Enter {str_brand('setWebrag True')} or {str_in_color('setWebrag False', Color.RED)}."
                        )

                elif cmd in ("", "rag"):
                    self.cli_ception()

                else:
                    print(f"Unknown command: {cmd}")
                    if " " in cmd or cmd.endswith("?"):
                        print(
                            f"Looks like a question! Press {str_brand('Enter')} first to start asking questions about your documents."
                        )
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

    def cli_ception(self):
        self.init_config()
        if self.ragPP is None or self.modified:
            try:
                with Spinner():
                    self.initialize_ragpp()
            except MilvusException as e:
                print_in_color(f"Failed to open the document database: {e}", Color.RED)
                print(
                    f"A previous session may still be holding it. Run {str_brand('pkill -f milvus_lite/lib/milvus')} and try again."
                )
                return
            self.modified = False
            print_in_color("RAG pipeline ready! Ask your questions.", Color.MMORE)
        print(str_in_color("Type /bye to exit.\n", Color.GRAY))
        while True:
            prompt = (
                f"RAG WITH PRIVACY {PRIVACY_EMOJI} > "
                if self.privacy_graph is not None
                else "RAG > "
            )
            query = input(str_in_color(prompt, Color.RED, bold=True))
            if query == "/bye":
                print_in_color("Exiting the RAG CLI", Color.RED, True)
                break
            elif query == _CONTEXT_CMD:
                self.print_context()
            elif self.privacy_graph is not None:
                results, timings = self.ask_with_privacy(query)
                self.print_answer(results, timings)
            else:
                with Spinner():
                    results, timings = self.do_rag(query)
                self.print_answer(results, timings)

    def ask_with_privacy(self, query: str):
        """Same query, but the privacy graph answers and draws itself as it runs."""
        from mmore.privacy.ux import attach, detach

        with live_status() as status:
            attach(status)
            start, waiting = time.monotonic(), paused_seconds()
            try:
                results, timings = self.do_rag(query)
            finally:
                detach()

        timings.total_time = time.monotonic() - start - (paused_seconds() - waiting)
        return results, timings

    def set_privacy(self, cmd: str) -> None:
        """`privacy` / `privacy on` / `privacy off` / `privacy <config.yaml>`."""
        arg = cmd.removeprefix("privacy").strip().lower()
        if not arg:
            state = str_brand("on", bold=True) if self.privacy_on else "off"
            print(f"Privacy mode is {state}. Use 'privacy on' or 'privacy off'.")
            return
        if arg == "off":
            if self.privacy_on:
                self.privacy_on = False
                self.modified = True  # the pipeline has to be rebuilt without it
            print("Privacy mode is off: the answer model sees the raw context.")
            return

        path = self.privacy_config_file if arg == "on" else cmd.split(" ", 1)[1].strip()
        if path is None:
            print_in_color(
                "No privacy config to enable. Use 'privacy <path to a privacy YAML>'.",
                Color.RED,
            )
            return
        from mmore.privacy.runner import load_privacy_config

        try:
            config = load_privacy_config(path)
        except Exception as e:
            print_in_color(f"Invalid privacy config {path}: {e}", Color.RED)
            return
        if path != self.privacy_config_file or self.privacy_config is None:
            self.privacy_config = config
        self.privacy_config_file = path
        self.privacy_on = True
        self.modified = True
        print(
            f"{PRIVACY_EMOJI} Privacy mode is {str_brand('on', bold=True)}: sensitive "
            f"data is detected and sanitized before the answer model sees it."
        )

    def print_privacy_config(self) -> None:
        """The privacy lines of `config`, only meaningful in privacy mode."""
        config = self.load_privacy_config() if self.privacy_on else None
        if config is None:
            return
        adversary = "on" if config.leakage_adversary.enabled else "off"
        gate = "on" if config.interactive else "off"
        print(f"leak adversary: {str_in_color(adversary, Color.BLUE)}")
        print(f"hitl approval gate: {str_in_color(gate, Color.BLUE)}")

    def print_context(self) -> None:
        """To print the raw context with command `/context`."""
        result = self.last_result or {}
        report = result.get("privacy_report")
        sanitized = result.get(PRIVACY_CHUNKS_KEY)
        if not report or sanitized is None:
            print(
                "No sanitized context yet: ask a question with privacy mode on first."
            )
            return
        from mmore.privacy.runner import render_chunks

        query = {
            "raw": result.get("input", ""),
            "sanitized": report.get("sanitized_query", ""),
        }
        chunks = [
            {"raw": doc.get("page_content", ""), "sanitized": text}
            for doc, text in zip(result.get("docs") or [], sanitized)
        ]
        print()
        print(render_chunks(query, chunks))
        print()

    def read_privacy_toggle(
        self, cmd: str, name: str
    ) -> Optional[Tuple[str, PrivacyConfig]]:
        arg = cmd.removeprefix(name).strip().lower()
        if arg not in ("", "on", "off"):
            print(f"Usage: '{name} on' or '{name} off'.")
            return None
        if not self.privacy_on:
            print_in_color(
                f"'{name}' sets up the privacy pipeline. Use 'privacy on' first.",
                Color.RED,
            )
            return None
        config = self.load_privacy_config()
        return None if config is None else (arg, config)

    def set_adversary(self, cmd: str) -> None:
        toggle = self.read_privacy_toggle(cmd, "adversary")
        if toggle is None:
            return
        arg, config = toggle
        adversary = config.leakage_adversary
        if not arg:
            state = str_brand("on", bold=True) if adversary.enabled else "off"
            print(f"The leak adversary is {state}. Use 'adversary on' or 'off'.")
            return

        enabled = arg == "on"
        if enabled != adversary.enabled:
            self.privacy_config = replace(
                config, leakage_adversary=replace(adversary, enabled=enabled)
            )
            self.modified = True  # the graph carries the setting, so rebuild it
        if enabled:
            print(
                f"The leak adversary is {str_brand('on', bold=True)}: it attacks the "
                f"sanitized context and sanitizes again ({adversary.max_iterations}x "
                f"at most) while it can still recover something."
            )
        else:
            print("The leak adversary is off: the context goes straight to the gate.")

    def set_hitl(self, cmd: str) -> None:
        toggle = self.read_privacy_toggle(cmd, "hitl")
        if toggle is None:
            return
        arg, config = toggle
        if not arg:
            state = str_brand("on", bold=True) if config.interactive else "off"
            print(f"The approval gate is {state}. Use 'hitl on' or 'off'.")
            return

        enabled = arg == "on"
        if enabled != config.interactive:
            self.privacy_config = replace(config, interactive=enabled)
            self.modified = True  # the gate reads the setting, so rebuild the graph
        if enabled:
            print(
                f"The approval gate is {str_brand('on', bold=True)}: every answer "
                f"waits for you to approve the sanitized context."
            )
        else:
            print(
                f"The approval gate is off: the sanitized context goes to the answer "
                f"model without asking. Check it afterwards with "
                f"{str_brand(_CONTEXT_CMD)}."
            )

    def load_privacy_config(self) -> Optional[PrivacyConfig]:
        """The session's privacy config, read from its path on first use."""
        if self.privacy_config is not None or self.privacy_config_file is None:
            return self.privacy_config
        from mmore.privacy.runner import load_privacy_config

        try:
            self.privacy_config = load_privacy_config(self.privacy_config_file)
        except Exception as e:
            print_in_color(
                f"Invalid privacy config {self.privacy_config_file}: {e}", Color.RED
            )
        return self.privacy_config

    def init_config(self):
        if self.ragConfig is None:
            self.ragConfig = load_config(self.config_file, RAGInferenceConfig)

    def initialize_ragpp(self):
        logger.info("Creating the RAG Pipeline...")
        # called only right after init_config
        assert self.ragConfig is not None
        self.privacy_graph, self.privacy_approver = None, None
        privacy_config = self.load_privacy_config() if self.privacy_on else None
        if privacy_config is not None:
            from mmore.privacy.runner import setup_privacy

            self.privacy_graph, self.privacy_approver, _ = setup_privacy(
                privacy_config, review_card=False
            )
        self.ragPP = RAGPipeline.from_config(
            self.ragConfig.rag,
            privacy_graph=self.privacy_graph,
            privacy_approver=self.privacy_approver,
        )
        logger.info("RAG pipeline initialized!")

    @profile_function()
    def do_rag(self, query) -> tuple[List[Dict[str, Any]], "TimingHandler"]:
        queries = [{"input": query, "collection_name": "my_docs"}]
        # called only after init_config and initialize_ragpp
        assert self.ragConfig is not None
        assert self.ragPP is not None

        timings = TimingHandler()
        results = self.ragPP(queries, return_dict=True, config={"callbacks": [timings]})
        return results, timings

    def print_answer(
        self, results: List[Dict[str, Any]], timings: Optional["TimingHandler"] = None
    ) -> None:
        assert self.ragConfig is not None
        assert len(results) == 1

        self.last_result = results[0]
        answer = results[0]["answer"].split("<|end_header_id|>")[-1].strip()
        print(f"\n{answer}\n")
        self._print_verifier(results[0])
        self._print_privacy(results[0])
        if timings is not None:
            self._print_metrics(results[0], answer, timings)
        self._print_context_hint(results[0])
        if self.ragConfig.rag.retriever.use_web:
            print("Sources:")
            for i in range(self.ragConfig.rag.retriever.k):
                url = results[0]["docs"][i]["metadata"]["url"]  # pyright: ignore
                title = results[0]["docs"][i]["metadata"]["title"]  # pyright: ignore
                print(f"  - {title}: {url}")
            print()

    def _print_verifier(self, result: Dict[str, Any]) -> None:
        """What the verifier checked on the answer above, and what it found."""
        report = result.get("privacy_report")
        if not report:
            return
        warnings = report["verifier_warnings"]
        failed = report["verifier_checks_failed"]
        checked = report["verifier_checks_run"]
        if not warnings and not failed and not checked:
            return  # the verifier ran no check, so it has nothing to report

        color = Color.ORANGE if warnings or failed else Color.GREEN
        bar = str_in_color("│", color)
        lines = [f"{bar} {str_in_color('Verifier report', color, bold=True)}"]
        if not warnings and checked:
            names = _join([check.replace("_", " ") for check in checked])
            lines.append(
                f"{bar} checked {names} against the raw context: no issue found"
            )
        for warning in warnings:
            kind = WarningKind(warning["kind"])
            finding = _VERIFIER_FINDINGS[kind]
            if kind is WarningKind.RESIDUAL_LEAKAGE and warning["entity_type"]:
                finding = f"{finding} ({warning['entity_type']})"
            detail = f"confidence {warning['confidence']:.2f}"
            if warning["count"] > 1:
                detail = f"{plural(warning['count'], 'flag')}, {detail}"
            lines.append(f"{bar} {finding}  {str_in_color(detail, Color.GRAY)}")
        for check in failed:
            name = check.replace("_", " ")
            lines.append(
                f"{bar} {str_in_color(f'the {name} check could not run', Color.GRAY)}"
            )
        print("\n".join(lines) + "\n")

    def _print_context_hint(self, result: Dict[str, Any]) -> None:
        """Point at the command that shows the documents behind the answer."""
        if not result.get("privacy_report"):
            return
        print(
            str_in_color("Use command ", Color.GRAY)
            + str_brand(_CONTEXT_CMD)
            + str_in_color(" to review your sanitized documents\n", Color.GRAY)
        )

    def _print_privacy(self, result: Dict[str, Any]) -> None:
        """One line on what the privacy pipeline did for this answer."""
        report = result.get("privacy_report")
        if not report:
            return

        engine = tool_name(report["detection_engine"])
        strategy = tool_name(report["sanitization_strategy"])
        parts = [
            f"{engine} (detector) + {strategy} (sanitizer)",
            plural(report["detection"]["count"], "sensitive span"),
        ]
        if report["adversary_iterations"]:
            parts.append(plural(report["adversary_iterations"], "leak escalation"))
        if report["human_iterations"]:
            parts.append(plural(report["human_iterations"], "revision"))
        agents = _privacy_seconds(report)
        if agents:
            parts.append(f"privacy {agents:.1f}s")
        _print_stats(parts)

    def _print_metrics(
        self, result: Dict[str, Any], answer: str, timings: "TimingHandler"
    ) -> None:
        assert self.ragConfig is not None
        llm = self.ragConfig.rag.llm

        report = result.get("privacy_report")
        if report:  # privacy mode answers with its own model, not the RAG one
            model = [f"{report['answer_model']} ({report['answer_backend']})"]
        else:
            model = [f"{llm.llm_name} ({'local' if llm.provider == 'HF' else 'API'})"]

        docs = result.get("docs") or []
        retrieved = [f"{len(docs)} chunks"]
        scores = [
            d["metadata"]["similarity"]
            for d in docs
            if d.get("metadata", {}).get("similarity") is not None
        ]
        if scores:
            retrieved.append(f"top score {max(scores):.2f}")

        timed = []
        if timings.retrieval_time is not None:
            timed.append(f"retrieval {timings.retrieval_time:.2f}s")
        if timings.generation_time is not None:
            label = "answer" if report else "generation"
            timed.append(f"{label} {timings.generation_time:.2f}s")
        if timings.total_time is not None:
            timed.append(f"total {timings.total_time:.2f}s")

        tokens = []
        ctx_tokens = self._count_tokens(result.get("context"))
        if ctx_tokens is not None:
            ctx = f"{ctx_tokens / 1000:.1f}k" if ctx_tokens >= 1000 else str(ctx_tokens)
            tokens.append(f"{ctx} context tokens")

        gen_tokens = timings.completion_tokens or self._count_tokens(answer)
        if gen_tokens:
            part = f"{gen_tokens} tokens"
            if timings.generation_time:
                part += f" @ {gen_tokens / timings.generation_time:.0f} tok/s"
            tokens.append(part)

        if report:
            _print_stats(model + retrieved + timed + tokens)
        else:
            _print_stats(model + timed)
            _print_stats(retrieved + tokens)
        print()

    def _count_tokens(self, text: Optional[str]) -> Optional[int]:
        """Token count using the local model tokenizer, if one is available."""
        if not text or self.ragPP is None:
            return None
        tokenizer = getattr(self.ragPP.llm, "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            return len(tokenizer.encode(text))
        except Exception:
            return None


def is_valid_model_path(model_path: str):
    try:
        model_info(model_path)
        return True, f"New model set to {str_in_color(model_path, Color.BLUE, True)}"
    except HfHubHTTPError as e:
        return (
            False,
            f"{str_in_color('There seems to be an error. Are you sure the model you are asking for exists?', Color.RED, True)} The error message: {e}",
        )


if __name__ == "__main__":
    quiet_noisy_libs(hide_info=True)
    enable_profiling_from_env()
    # example usage: python -m mmore.ragcli --config-file examples/rag/config.yaml
    # optional flag: --privacy examples/rag/privacy.yml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", required=True, help="Path to the RAG configuration file."
    )
    parser.add_argument(
        "--privacy",
        default=None,
        help="Path to a privacy config to start the chat in privacy mode.",
    )
    args = parser.parse_args()

    my_rag_cli = RagCLI(args.config_file, privacy_config_file=args.privacy)
    my_rag_cli.launch_cli()
