"""Connects retrieval and answer generation into the complete RAG
workflow: load vector store -> retrieve context -> build prompt ->
generate answer -> return answer with traceable sources."""

import os
from typing import Dict

from src.embeddings import HFEmbeddings
from src.vector_database import VectorDatabase
from src.retriever import Retriever
from src.prompt import build_prompt
from src.llm import HFLLM

VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")
TOP_K = int(os.getenv("TOP_K", "4"))


class RAGPipeline:
    def __init__(self):
        self.embeddings = HFEmbeddings()
        self.vector_db = VectorDatabase(
            dimension=self.embeddings.dimension,
            index_path=os.path.join(VECTOR_STORE_DIR, "index.faiss"),
            meta_path=os.path.join(VECTOR_STORE_DIR, "meta.pkl"),
        )

        if self.vector_db.exists():
            self.vector_db.load()
        else:
            print("[rag_pipeline] No existing vector store found. Run `python ingest.py` first.")

        self.retriever = Retriever(self.embeddings, self.vector_db, top_k=TOP_K)
        self.llm = HFLLM()

    def answer(self, question: str) -> Dict:
        contexts = self.retriever.retrieve(question)

        if not contexts:
            return {
                "answer": "I don't have enough information in the documents to answer that.",
                "sources": [],
                "contexts": [],
            }

        prompt = build_prompt(question, contexts)
        answer_text = self.llm.generate(prompt)
        sources = sorted({os.path.basename(str(c.get("source"))) for c in contexts})

        return {"answer": answer_text, "sources": sources, "contexts": contexts}
