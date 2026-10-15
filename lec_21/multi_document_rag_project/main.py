"""Interactive terminal application for the multi-document RAG system.

Run:
    python main.py

Supports slash commands (type /help in the prompt):
    /help /commands /ingest /list /stats /sources /eval /reset /exit
"""

import contextlib
import io
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import commands
from commands import is_command, parse_command, render_help
from src.rag_pipeline import RAGPipeline


RESET = "\033[0m"
CYAN = "\033[96m"
BLUE = "\033[94m"
WHITE = "\033[97m"
MUTED = "\033[90m"
GREEN = "\033[92m"
YELLOW = "\033[93m"

VECTOR_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")


def paint(text: str, color: str) -> str:
    """Apply terminal color when output is connected to an interactive shell."""
    if not os.isatty(1):
        return text
    return f"{color}{text}{RESET}"


def print_header() -> None:
    print()
    print(paint("  KNOWLEDGE DESK", CYAN))
    print(paint("  Multi-document question answering", MUTED))
    print(paint("  " + "─" * 46, BLUE))
    print(paint("  Type /help for commands, or ask a question.", MUTED))
    print(paint("  " + "─" * 46, BLUE))
    print()


def load_pipeline() -> RAGPipeline:
    print(paint("  ◌  Loading your knowledge desk...", MUTED))
    # Transformers and Hugging Face print download progress during startup.
    # Keep that implementation detail out of the user-facing interface.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        pipeline = RAGPipeline()
    print(paint("  ●  Ready", GREEN))
    print()
    return pipeline


# ---------------------------------------------------------------------------
# Command handlers (terminal versions)
# ---------------------------------------------------------------------------


def handle_command(name: str, arg: str, pipeline: RAGPipeline) -> bool:
    """Execute a slash command. Returns False when the app should exit."""
    if name in ("/help", "/commands"):
        print()
        print(paint(render_help(), BLUE))
        print()
        return True

    if name == "/ingest":
        print(paint("  ◌  Rebuilding the vector store from data/…", MUTED))
        import ingest

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            stats = ingest.main()
        print(paint("  ●  Index rebuilt", GREEN))
        print()
        return True

    if name == "/list":
        stats = commands.index_stats(VECTOR_DIR)
        if not stats:
            print(paint("  No index yet — run /ingest first.", YELLOW))
        else:
            print()
            print(paint(f"  {stats['total_chunks']} chunks across "
                        f"{len(stats['sources'])} document(s):", BLUE))
            for src, count in stats["sources"].items():
                print(paint(f"    • {src} — {count} chunk(s)", MUTED))
            print()
        return True

    if name == "/stats":
        stats = commands.index_stats(VECTOR_DIR)
        if not stats:
            print(paint("  No index yet — run /ingest first.", YELLOW))
        else:
            print()
            print(paint("  INDEX STATS", BLUE))
            print(paint(f"    Total chunks        : {stats['total_chunks']}", MUTED))
            print(paint(f"    Documents           : {len(stats['sources'])}", MUTED))
            dim = stats["dimension"] or "?"
            print(paint(f"    Embedding dimension : {dim}", MUTED))
            print(paint(f"    Sources             : {', '.join(stats['sources'])}", MUTED))
            print()
        return True

    if name == "/sources":
        last = getattr(handle_command, "_last_sources", [])
        if not last:
            print(paint("  No previous answer to show sources for.", MUTED))
        else:
            print()
            print(paint("  SOURCES (last answer)", BLUE))
            for i, src in enumerate(last, 1):
                print(paint(f"    {i:>2}  {src}", MUTED))
            print()
        return True

    if name == "/eval":
        print(paint("  ◌  Running RAGAS evaluation…", MUTED))
        try:
            import subprocess
            import sys

            subprocess.run([sys.executable, "-m", "evaluation.ragas_evaluation"], check=False)
        except Exception as error:
            print(paint(f"  Evaluation failed: {error}", YELLOW))
        return True

    if name == "/reset":
        handle_command._last_sources = []
        print(paint("  ●  Conversation cleared", GREEN))
        print()
        return True

    if name in ("/exit", "/quit"):
        return False

    print(paint("  " + commands.unknown_command_msg(name), YELLOW))
    print()
    return True


def main():
    print_header()

    try:
        pipeline = load_pipeline()
    except Exception as error:
        print(paint(f"  Could not start: {error}", YELLOW))
        print(paint("  Check the setup instructions in README.md.", MUTED))
        return

    while True:
        try:
            question = input(paint("  You  ", CYAN) + paint("› ", WHITE)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue

        # Slash commands
        if is_command(question):
            name, arg = parse_command(question)
            if not handle_command(name, arg, pipeline):
                break
            continue

        # Regular question
        result = pipeline.answer(question)

        print()
        print(paint("  ANSWER", BLUE))
        print(paint("  " + "─" * 46, MUTED))
        print(f"  {result['answer']}")
        print()
        print(paint("  SOURCES", BLUE))
        if result["sources"]:
            for i, src in enumerate(result["sources"], start=1):
                print(paint(f"  {i:>2}  {src}", MUTED))
        else:
            print(paint("  No matching sources found.", MUTED))
        print()

        handle_command._last_sources = result["sources"]


if __name__ == "__main__":
    main()
