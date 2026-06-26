import argparse
from typing import Any, Dict, List, Optional

from huggingface_hub import model_info
from huggingface_hub.errors import HfHubHTTPError
from pymilvus.exceptions import MilvusException

from mmore.profiler import enable_profiling_from_env, profile_function
from mmore.rag.pipeline import RAGPipeline
from mmore.ragcli_helper import TimingHandler
from mmore.run_rag import RAGInferenceConfig
from mmore.utils import load_config
from mmore.ux import (
    Color,
    Spinner,
    print_in_color,
    quiet_noisy_libs,
    setup_logging,
    str_brand,
    str_in_color,
)

RAG_NAME = "RAG"
RAG_EMOJI = "🧠"
logger = setup_logging(RAG_NAME, RAG_EMOJI)


class RagCLI:
    def __init__(self, config_file: str):
        self.ragConfig: Optional[RAGInferenceConfig] = None
        self.ragPP = None
        self.modified: bool = (
            False  # flag to indicate if the configuration has been modified
        )
        self.config_file = config_file

    def launch_cli(self):
        quiet_noisy_libs(hide_info=True)
        print_in_color(
            "Welcome to this RAG command-line interface! 🧠", Color.BRAND, bold=True
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
                    elif command == "exit":
                        print("Exit the CLI.")
                    else:
                        print("Sorry, this command does not exist.")

                elif cmd == "config":
                    self.init_config()
                    assert self.ragConfig is not None

                    confrag = self.ragConfig.rag
                    print(
                        f"k: {str_in_color(confrag.retriever.k, Color.BLUE)} \nmodel: {str_in_color(confrag.llm.llm_name, Color.BLUE)} \nuse web for rag: {str_in_color(confrag.retriever.use_web, Color.BLUE)}"
                    )
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
            print_in_color("RAG pipeline ready! Ask your questions.", Color.BRAND)
        print(str_in_color("Type /bye to exit.\n", Color.GRAY))
        while True:
            query = input(str_in_color("RAG > ", Color.RED, bold=True))
            if query == "/bye":
                print_in_color("Exiting the RAG CLI", Color.RED, True)
                break
            else:
                with Spinner():
                    results, timings = self.do_rag(query)
                self.print_answer(results, timings)

    def init_config(self):
        if self.ragConfig is None:
            self.ragConfig = load_config(self.config_file, RAGInferenceConfig)

    def initialize_ragpp(self):
        logger.info("Creating the RAG Pipeline...")
        # called only right after init_config
        assert self.ragConfig is not None
        self.ragPP = RAGPipeline.from_config(self.ragConfig.rag)
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

        answer = results[0]["answer"].split("<|end_header_id|>")[-1].strip()
        print(f"\n{answer}\n")
        if timings is not None:
            self._print_metrics(results[0], answer, timings)
        if self.ragConfig.rag.retriever.use_web:
            print("Sources:")
            for i in range(self.ragConfig.rag.retriever.k):
                url = results[0]["docs"][i]["metadata"]["url"]  # pyright: ignore
                title = results[0]["docs"][i]["metadata"]["title"]  # pyright: ignore
                print(f"  - {title}: {url}")
            print()

    def _print_metrics(
        self, result: Dict[str, Any], answer: str, timings: "TimingHandler"
    ) -> None:
        assert self.ragConfig is not None
        llm = self.ragConfig.rag.llm

        line1 = [f"{llm.llm_name} ({'local' if llm.provider == 'HF' else 'API'})"]
        if timings.retrieval_time is not None:
            line1.append(f"retrieval {timings.retrieval_time:.2f}s")
        if timings.generation_time is not None:
            line1.append(f"generation {timings.generation_time:.2f}s")

        docs = result.get("docs") or []
        line2 = [f"{len(docs)} chunks"]

        ctx_tokens = self._count_tokens(result.get("context"))
        if ctx_tokens is not None:
            ctx = f"{ctx_tokens / 1000:.1f}k" if ctx_tokens >= 1000 else str(ctx_tokens)
            line2.append(f"{ctx} context tokens")

        gen_tokens = timings.completion_tokens or self._count_tokens(answer)
        if gen_tokens:
            part = f"{gen_tokens} tokens"
            if timings.generation_time:
                part += f" @ {gen_tokens / timings.generation_time:.0f} tok/s"
            line2.append(part)

        scores = [
            d["metadata"]["similarity"]
            for d in docs
            if d.get("metadata", {}).get("similarity") is not None
        ]
        if scores:
            line2.append(f"top score {max(scores):.2f}")

        print(str_in_color(" | ".join(line1), Color.GRAY))
        print(str_in_color(" | ".join(line2), Color.GRAY) + "\n")

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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", required=True, help="Path to the RAG configuration file."
    )
    args = parser.parse_args()

    my_rag_cli = RagCLI(args.config_file)
    my_rag_cli.launch_cli()
