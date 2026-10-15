"""Shared slash-command layer for the multi-document RAG system.

Both the terminal app (main.py) and the Streamlit app (app.py) route user
input through this module so the command surface stays identical everywhere:

    /help  /commands  /ingest  /list  /stats  /sources  /eval  /reset  /exit
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Command registry — single source of truth for menus and routers
# ---------------------------------------------------------------------------

COMMANDS: List[Dict[str, str]] = [
    {"name": "/help", "args": "", "desc": "Show the help menu"},
    {"name": "/commands", "args": "", "desc": "List every available command"},
    {
        "name": "/ingest",
        "args": "",
        "desc": "Rebuild the vector store from data/ (plus uploaded files in the web app)",
    },
    {"name": "/list", "args": "", "desc": "List indexed documents with chunk counts"},
    {"name": "/stats", "args": "", "desc": "Index statistics: chunks, sources, dimensions"},
    {"name": "/sources", "args": "", "desc": "Show sources behind the most recent answer"},
    {
        "name": "/eval",
        "args": "",
        "desc": "Run the RAGAS evaluation suite (needs HF_TOKEN, see .env.example)",
    },
    {"name": "/reset", "args": "", "desc": "Clear the conversation history"},
    {"name": "/exit", "args": "", "desc": "Leave the app (terminal only; web app ignores it)"},
]

COMMAND_NAMES = [c["name"] for c in COMMANDS]


def is_command(text: str) -> bool:
    """A command is any token starting with '/' entered on its own."""
    return text.strip().startswith("/")


def parse_command(text: str) -> Tuple[Optional[str], str]:
    """Split raw input into (command_name, argument). Returns (None, text)
    when the input is not a command."""
    stripped = text.strip()
    if not is_command(stripped):
        return None, stripped
    parts = stripped.split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""
    return name, arg


def render_help(title: str = "Available commands") -> str:
    """Plain-text help table shared by the terminal menu and web chat."""
    lines = [title, ""]
    width = max(len(c["name"]) for c in COMMANDS)
    for c in COMMANDS:
        args = f" {c['args']}" if c["args"] else ""
        lines.append(f"  {c['name']}{args:<{width - len(c['name']) + 2}} — {c['desc']}")
    lines += ["", "Anything that doesn't start with '/' is treated as a question."]
    return "\n".join(lines)


def unknown_command_msg(name: str) -> str:
    return (
        f"Unknown command: {name}\n"
        f"Type /commands to see everything I can do "
        f"({', '.join(COMMAND_NAMES)})."
    )


# ---------------------------------------------------------------------------
# Index inspection helpers (shared by CLI commands and diagnostics)
# ---------------------------------------------------------------------------


def load_index_metadata(vector_store_dir: str) -> List[Dict]:
    meta_path = os.path.join(vector_store_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        return []
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def index_stats(vector_store_dir: str) -> Optional[Dict]:
    """Summarize what is inside a vector store directory."""
    metadata = load_index_metadata(vector_store_dir)
    if not metadata:
        return None

    per_source: Dict[str, int] = {}
    for meta in metadata:
        src = os.path.basename(str(meta.get("source")))
        per_source[src] = per_source.get(src, 0) + 1

    dim = None
    index_path = os.path.join(vector_store_dir, "index.faiss")
    if os.path.exists(index_path):
        try:
            import faiss

            dim = faiss.read_index(index_path).d
        except Exception:
            dim = None

    return {
        "total_chunks": len(metadata),
        "sources": dict(sorted(per_source.items())),
        "dimension": dim,
    }
