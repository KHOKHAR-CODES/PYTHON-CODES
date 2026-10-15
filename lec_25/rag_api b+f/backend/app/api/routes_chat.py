
from pathlib import Path

from fastapi import APIRouter, Request

from app.core.config import get_settings
from app.core.exceptions import InvalidRequestError
from app.schemas.request import AttachDocumentRequest, ChatRequest, SummarizeHistoryRequest
from app.schemas.response import (
    ChatResponse, ErrorResponse, HistoryResponse, SessionDocumentResponse,
    SessionDocumentsResponse, SummaryResponse, ChatMessage, DocumentInfo,
)

router = APIRouter(tags=['conversational-rag'])


def _validate_session(session_id: str) -> None:
    # Pydantic validates syntax; this explicit check keeps direct service calls safe.
    if not session_id.strip():
        raise InvalidRequestError('session_id must not be empty.')


@router.post('/chat', response_model=ChatResponse, responses={400: {'model': ErrorResponse}, 422: {'model': ErrorResponse}, 502: {'model': ErrorResponse}, 503: {'model': ErrorResponse}, 504: {'model': ErrorResponse}}, summary='Answer a question using session-aware conversational RAG')
def chat(request: Request, payload: ChatRequest):
    settings = get_settings()
    _validate_session(payload.session_id)
    if payload.top_k > settings.max_top_k:
        raise InvalidRequestError(f'top_k must be <= {settings.max_top_k}.')
    if len(payload.question) > settings.max_question_length:
        raise InvalidRequestError(f'question must be <= {settings.max_question_length} characters.')
    result = request.app.state.chat_service.chat(
        payload.session_id, payload.question, payload.top_k, payload.similarity_threshold
    )
    return ChatResponse(
        request_id=request.state.request_id,
        session_id=payload.session_id,
        question=payload.question,
        answer=result['answer'],
        sources=result['sources'] if payload.include_sources else [],
        metadata={
            'retrieved_chunks': result['retrieved_chunks'],
            'selected_chunks': result['selected_chunks'],
            'model': result['model'],
            'latency_ms': result['latency_ms'],
            'embedding_latency_ms': result['embedding_latency_ms'],
            'retrieval_latency_ms': result['retrieval_latency_ms'],
            'llm_latency_ms': result['llm_latency_ms'],
            'history_messages': result['history_messages'],
            'session_documents': result['session_documents'],
            'summary_used': result['summary_used'],
        },
    )


@router.post('/attach-document', response_model=SessionDocumentResponse, responses={400: {'model': ErrorResponse}}, summary='Index and attach a local document to a chat session')
def attach_document(request: Request, payload: AttachDocumentRequest):
    settings = get_settings()
    if len(payload.file_path) > settings.max_file_path_length:
        raise InvalidRequestError('file_path is too long.')
    request.app.state.history_service.ensure_session(payload.session_id)
    result = request.app.state.document_service.index_file(
        payload.file_path,
        allowed_roots=['./data', './uploads'],
        allowed_extensions={x.strip().lower() for x in settings.allowed_file_extensions.split(',') if x.strip()},
    )
    try:
        request.app.state.session_manager.attach_document(
            payload.session_id, result['document_id'],
            __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
        )
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc
    return SessionDocumentResponse(**result, attached=True)


@router.get('/history/{session_id}', response_model=HistoryResponse, responses={400: {'model': ErrorResponse}}, summary='Get conversation history for a session')
def get_history(request: Request, session_id: str):
    _validate_session(session_id)
    request.app.state.history_service.ensure_session(session_id)
    summary = request.app.state.history_service.get_summary(session_id)
    messages = request.app.state.history_service.get_messages(session_id)
    return HistoryResponse(
        session_id=session_id,
        summary=summary,
        messages=[ChatMessage(**m) for m in messages],
    )


@router.delete('/history/{session_id}', response_model=HistoryResponse, responses={400: {'model': ErrorResponse}}, summary='Clear conversation history without deleting session documents')
def clear_history(request: Request, session_id: str):
    _validate_session(session_id)
    request.app.state.history_service.clear(session_id)
    request.app.state.history_service.ensure_session(session_id)
    return HistoryResponse(session_id=session_id, summary=None, messages=[])


@router.post('/summarize-history', response_model=SummaryResponse, responses={400: {'model': ErrorResponse}, 502: {'model': ErrorResponse}, 504: {'model': ErrorResponse}}, summary='Summarize a session conversation for context-window control')
def summarize_history(request: Request, payload: SummarizeHistoryRequest):
    _validate_session(payload.session_id)
    result = request.app.state.chat_service.summarize(payload.session_id)
    return SummaryResponse(**result)


@router.get('/session/{session_id}/documents', response_model=SessionDocumentsResponse, responses={400: {'model': ErrorResponse}}, summary='List documents attached to a chat session')
def session_documents(request: Request, session_id: str):
    _validate_session(session_id)
    ids = request.app.state.session_manager.document_ids(session_id)
    all_docs = request.app.state.vector_store.list_documents()
    id_set = set(ids)
    docs = [DocumentInfo(**doc) for doc in all_docs if doc['document_id'] in id_set]
    return SessionDocumentsResponse(session_id=session_id, documents=docs)
