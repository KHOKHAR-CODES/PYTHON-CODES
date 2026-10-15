"""Builds the final prompt sent to the LLM from the user question and
the retrieved context chunks."""

from typing import List, Dict

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. If the answer is not contained in the context, say "
    "you don't have enough information. Be concise."
)


def build_prompt(question: str, contexts: List[Dict]) -> str:
    context_block = "\n\n".join(
        f"[Source: {c.get('source')}" + (f", page {c.get('page')}" if c.get("page") else "") + f"]\n{c.get('text')}"
        for c in contexts
    )
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
