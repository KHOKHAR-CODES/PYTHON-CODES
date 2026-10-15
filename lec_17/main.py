"""Interactive terminal application for the multi-document RAG system.

Run:
    python main.py
"""

import contextlib
import io
import os

from dotenv import load_dotenv

load_dotenv()

from src.rag_pipeline import RAGPipeline


RESET = "\033[0m"
CYAN = "\033[96m"
BLUE = "\033[94m"
WHITE = "\033[97m"
MUTED = "\033[90m"
GREEN = "\033[92m"
YELLOW = "\033[93m"


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
    print(paint("  Type a question, or 'exit' to close.", MUTED))
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
        if question.lower() in {"exit", "quit"}:
            break

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


if __name__ == "__main__":
    main()
