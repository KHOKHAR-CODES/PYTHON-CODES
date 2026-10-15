
from __future__ import annotations

import time

from app.core.exceptions import NoRelevantContextError
from app.infrastructure.llm_client import LLMClient
from app.services.embedding_service import EmbeddingService
from app.services.history_service import HistoryService
from app.services.prompt_service import SYSTEM_PROMPT, build_context, build_conversational_prompt
from app.services.search_service import SearchService
from app.services.session_manager import SessionManager


class ChatService:
    def __init__(
        self, embedding_service, search_service, llm_client, history_service, session_manager,
        max_top_k, default_top_k, include_source_previews, source_preview_chars,
        history_max_messages, history_max_chars, recent_messages_count, summary_threshold
    ):
        self.embedding_service = embedding_service
        self.search_service = search_service
        self.llm_client = llm_client
        self.history_service = history_service
        self.session_manager = session_manager
        self.max_top_k = max_top_k
        self.default_top_k = default_top_k
        self.include_source_previews = include_source_previews
        self.source_preview_chars = source_preview_chars
        self.history_max_messages = history_max_messages
        self.history_max_chars = history_max_chars
        self.recent_messages_count = recent_messages_count
        self.summary_threshold = summary_threshold

    def chat(self, session_id, question, top_k, similarity_threshold):
        started = time.perf_counter()
        top_k = self.default_top_k if top_k is None else min(top_k, self.max_top_k)
        self.history_service.ensure_session(session_id)
        if self.history_service.count_messages(session_id) >= self.summary_threshold and not self.history_service.get_summary(session_id):
            self.summarize(session_id)
        summary, history = self.history_service.build_context(
            session_id, self.history_max_messages, self.history_max_chars, self.recent_messages_count
        )
        document_ids = self.session_manager.document_ids(session_id)
        embedding, embedding_ms = self.embedding_service.embed_query(question)
        retrieval_started = time.perf_counter()
        filtered = self.search_service.retrieve(
            embedding, top_k, similarity_threshold, document_ids=document_ids
        )
        retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)

        if not filtered.selected:
            answer = NoRelevantContextError().message
            self.history_service.add_message(session_id, 'user', question)
            self.history_service.add_message(session_id, 'assistant', answer)
            return self._result(
                session_id, answer, [], len(filtered.retrieved), 0, started,
                embedding_ms, retrieval_ms, None, len(history), len(document_ids), bool(summary)
            )

        context = build_context(filtered.selected)
        prompt = build_conversational_prompt(context, question, history, summary)
        answer, llm_ms = self.llm_client.generate_answer(SYSTEM_PROMPT, prompt)
        self.history_service.add_message(session_id, 'user', question)
        self.history_service.add_message(session_id, 'assistant', answer)

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
        return self._result(
            session_id, answer, sources, len(filtered.retrieved), len(filtered.selected),
            started, embedding_ms, retrieval_ms, llm_ms, len(history), len(document_ids),
            bool(summary), [r.record.text for r in filtered.selected], [r.score for r in filtered.selected]
        )

    def _result(self, session_id, answer, sources, retrieved, selected, started,
                embedding_ms, retrieval_ms, llm_ms, history_count, document_count,
                summary_used, selected_texts=None, selected_scores=None):
        return {
            'session_id': session_id, 'answer': answer, 'sources': sources,
            'retrieved_chunks': retrieved, 'selected_chunks': selected,
            'model': self.llm_client.model_name,
            'latency_ms': int((time.perf_counter() - started) * 1000),
            'embedding_latency_ms': embedding_ms, 'retrieval_latency_ms': retrieval_ms,
            'llm_latency_ms': llm_ms, 'history_messages': history_count,
            'session_documents': document_count, 'summary_used': summary_used,
            'selected_texts': selected_texts or [], 'selected_scores': selected_scores or [],
        }

    def summarize(self, session_id):
        self.history_service.ensure_session(session_id)
        messages = self.history_service.get_messages(session_id)
        if not messages:
            return {'session_id': session_id, 'summary': '', 'summarized_messages': 0}
        history_text = '\n'.join(f'{m["role"].upper()}: {m["content"]}' for m in messages)
        prompt = (
            "Summarize the conversation for future question answering. Preserve important "
            "facts, definitions, unresolved questions, references, and user-provided constraints. "
            "Do not add facts. Return only the concise summary.\n\n" + history_text
        )
        summary, _ = self.llm_client.generate_answer(
            'You summarize a conversation faithfully for later retrieval. Do not invent information.',
            prompt,
        )
        self.history_service.set_summary(session_id, summary)
        return {'session_id': session_id, 'summary': summary, 'summarized_messages': len(messages)}
