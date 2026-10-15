from __future__ import annotations

import re
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'how',
    'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'what',
    'when', 'where', 'which', 'who', 'why', 'with', 'do', 'does', 'can',
}


@dataclass(frozen=True)
class EvaluationResult:
    relevance_score: float
    source_available: bool
    answer_supported: bool
    status: str


class EvaluationService:
    """Small, deterministic evaluator for retrieval and answer quality.

    It deliberately does not ask the LLM to grade its own output. Scores are
    derived from retrieval similarity, source presence, and lexical evidence
    in the retrieved context.
    """

    def __init__(self, retrieval_threshold: float):
        self.retrieval_threshold = retrieval_threshold

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token.lower()
            for token in _TOKEN_RE.findall(text or '')
            if token.lower() not in _STOPWORDS and len(token) > 1
        }

    def _answer_relevance(self, question: str, answer: str) -> float:
        question_tokens = self._tokens(question)
        answer_tokens = self._tokens(answer)
        if not question_tokens or not answer_tokens:
            return 0.0
        return len(question_tokens & answer_tokens) / len(question_tokens)

    def evaluate(
        self,
        question: str,
        answer: str,
        sources: list[str],
        retrieved_chunks: list[str] | None = None,
        similarity_scores: list[float] | None = None,
    ) -> EvaluationResult:
        source_available = any(source.strip() for source in sources)
        chunks = [chunk.strip() for chunk in (retrieved_chunks or []) if chunk and chunk.strip()]
        scores = [max(0.0, min(1.0, float(score))) for score in (similarity_scores or [])]

        if scores:
            retrieval_relevance = max(scores)
        elif chunks:
            retrieval_relevance = 1.0
        else:
            retrieval_relevance = 0.0

        answer_relevance = self._answer_relevance(question, answer)
        context_text = ' '.join(chunks)
        answer_tokens = self._tokens(answer)
        context_tokens = self._tokens(context_text)
        grounding_overlap = (
            len(answer_tokens & context_tokens) / len(answer_tokens)
            if answer_tokens and context_tokens else 0.0
        )

        # In the full RAG flow, support requires relevant retrieval, a source,
        # and enough lexical evidence that the answer is grounded in context.
        if chunks:
            combined_relevance = (retrieval_relevance + answer_relevance) / 2.0
            answer_supported = (
                source_available
                and retrieval_relevance >= self.retrieval_threshold
                and grounding_overlap >= 0.20
                and combined_relevance >= self.retrieval_threshold
            )
            relevance_score = max(0.0, min(1.0, combined_relevance))
        else:
            # /evaluate receives no chunks, so evaluate the supplied answer
            # independently while never claiming grounding without sources.
            relevance_score = max(0.0, min(1.0, answer_relevance))
            answer_supported = source_available and answer_relevance >= self.retrieval_threshold

        status = 'good' if answer_supported and relevance_score >= self.retrieval_threshold else 'insufficient context'
        return EvaluationResult(
            relevance_score=round(relevance_score, 4),
            source_available=source_available,
            answer_supported=answer_supported,
            status=status,
        )
