import time

from app.core.exceptions import NoRelevantContextError
from app.services.embedding_service import EmbeddingService
from app.services.prompt_service import SYSTEM_PROMPT, build_context, build_user_prompt
from app.services.search_service import SearchService
from app.infrastructure.llm_client import LLMClient


class RAGService:
    def __init__(self, embedding_service: EmbeddingService, search_service: SearchService, llm_client: LLMClient, max_top_k: int, default_top_k: int, include_source_previews: bool, source_preview_chars: int):
        self.embedding_service = embedding_service
        self.search_service = search_service
        self.llm_client = llm_client
        self.max_top_k = max_top_k
        self.default_top_k = default_top_k
        self.include_source_previews = include_source_previews
        self.source_preview_chars = source_preview_chars

    def ask(self, question: str, top_k: int | None, similarity_threshold: float | None) -> dict:
        started = time.perf_counter()
        top_k = self.default_top_k if top_k is None else min(top_k, self.max_top_k)
        embedding, embedding_ms = self.embedding_service.embed_query(question)
        retrieval_started = time.perf_counter()
        filtered = self.search_service.retrieve(embedding, top_k, similarity_threshold)
        retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)
        if not filtered.selected:
            return {
                'answer': NoRelevantContextError().message,
                'sources': [],
                'retrieved_chunks': len(filtered.retrieved),
                'selected_chunks': 0,
                'model': self.llm_client.model_name,
                'latency_ms': int((time.perf_counter() - started) * 1000),
                'embedding_latency_ms': embedding_ms,
                'retrieval_latency_ms': retrieval_ms,
                'llm_latency_ms': None,
                'selected_texts': [],
                'selected_scores': [],
            }
        context = build_context(filtered.selected)
        prompt = build_user_prompt(context, question)
        answer, llm_ms = self.llm_client.generate_answer(SYSTEM_PROMPT, prompt)
        sources = []
        for result in filtered.selected:
            record = result.record
            sources.append({
                'document': record.document_name,
                'document_id': record.document_id,
                'chunk_id': record.chunk_id,
                'score': round(result.score, 6),
                'page': record.page,
                'text_preview': record.text[:self.source_preview_chars] if self.include_source_previews else None,
            })
        return {
            'answer': answer,
            'sources': sources,
            'retrieved_chunks': len(filtered.retrieved),
            'selected_chunks': len(filtered.selected),
            'model': self.llm_client.model_name,
            'latency_ms': int((time.perf_counter() - started) * 1000),
            'embedding_latency_ms': embedding_ms,
            'retrieval_latency_ms': retrieval_ms,
            'llm_latency_ms': llm_ms,
            'selected_texts': [result.record.text for result in filtered.selected],
            'selected_scores': [result.score for result in filtered.selected],
        }
