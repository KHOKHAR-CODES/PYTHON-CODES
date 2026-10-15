from fastapi import APIRouter, Request

from app.core.config import get_settings
from app.core.exceptions import InvalidRequestError
from app.schemas.request import AskRequest, EvaluationRequest
from app.schemas.response import AskEvaluateResponse, AskResponse, ErrorResponse, EvaluationResponse

router = APIRouter(tags=['question-answering', 'evaluation'])


@router.post('/ask', response_model=AskResponse, responses={400: {'model': ErrorResponse}, 422: {'model': ErrorResponse}, 502: {'model': ErrorResponse}, 503: {'model': ErrorResponse}, 504: {'model': ErrorResponse}}, summary='Answer a question using indexed documents')
def ask(request: Request, payload: AskRequest):
    settings = get_settings()
    if payload.top_k > settings.max_top_k:
        raise InvalidRequestError(f'top_k must be <= {settings.max_top_k}.')
    if len(payload.question) > settings.max_question_length:
        raise InvalidRequestError(f'question must be <= {settings.max_question_length} characters.')
    result = request.app.state.rag_service.ask(payload.question, payload.top_k, payload.similarity_threshold)
    return AskResponse(
        request_id=request.state.request_id,
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
        },
    )


@router.post('/evaluate', response_model=EvaluationResponse, responses={400: {'model': ErrorResponse}, 422: {'model': ErrorResponse}}, summary='Evaluate an already-generated RAG response')
def evaluate(request: Request, payload: EvaluationRequest):
    result = request.app.state.evaluation_service.evaluate(
        question=payload.question,
        answer=payload.answer,
        sources=payload.sources,
    )
    return EvaluationResponse(**result.__dict__)


@router.post('/ask-evaluate', response_model=AskEvaluateResponse, responses={400: {'model': ErrorResponse}, 422: {'model': ErrorResponse}, 502: {'model': ErrorResponse}, 503: {'model': ErrorResponse}, 504: {'model': ErrorResponse}}, summary='Run the RAG pipeline and evaluate the generated answer')
def ask_evaluate(request: Request, payload: AskRequest):
    settings = get_settings()
    if payload.top_k > settings.max_top_k:
        raise InvalidRequestError(f'top_k must be <= {settings.max_top_k}.')
    if len(payload.question) > settings.max_question_length:
        raise InvalidRequestError(f'question must be <= {settings.max_question_length} characters.')

    result = request.app.state.rag_service.ask(payload.question, payload.top_k, payload.similarity_threshold)
    source_names = [source['document'] for source in result['sources']]
    evaluation = request.app.state.evaluation_service.evaluate(
        question=payload.question,
        answer=result['answer'],
        sources=source_names,
        retrieved_chunks=result['selected_texts'],
        similarity_scores=result['selected_scores'],
    )
    return AskEvaluateResponse(
        request_id=request.state.request_id,
        question=payload.question,
        answer=result['answer'],
        sources=result['sources'] if payload.include_sources else [],
        retrieved_chunks=result['retrieved_chunks'],
        evaluation=EvaluationResponse(**evaluation.__dict__),
    )
