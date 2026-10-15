import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_ask import router as ask_router
from app.api.routes_documents import router as documents_router
from app.api.routes_health import router as health_router
from app.api.routes_chat import router as chat_router
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, request_id_var
from app.infrastructure.embeddings import HashEmbeddingProvider, OpenAICompatibleEmbeddingProvider
from app.infrastructure.vector_store import SQLiteNumpyVectorStore
from app.infrastructure.llm_client import OpenAICompatibleLLMClient, TestLLMClient
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.rag_service import RAGService
from app.services.evaluation_service import EvaluationService
from app.services.search_service import SearchService
from app.services.history_service import HistoryService
from app.services.session_manager import SessionManager
from app.services.chat_service import ChatService
from app.utils.ids import new_request_id

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger('rag_api')


class InMemoryRateLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds
        self.buckets: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        bucket = self.buckets[key]
        while bucket and now - bucket[0] >= self.window:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return False
        bucket.append(now)
        return True


def build_components():
    settings.validate_runtime()
    if settings.embedding_provider == 'hash':
        embedding_provider = HashEmbeddingProvider(settings.embedding_dimension)
    else:
        embedding_provider = OpenAICompatibleEmbeddingProvider(
            settings.embedding_base_url, settings.embedding_api_key or '', settings.embedding_model,
            settings.embedding_dimension, settings.embedding_timeout_seconds, settings.retry_attempts, settings.retry_backoff_seconds,
        )
    embedding_service = EmbeddingService(embedding_provider)

    vector_store = SQLiteNumpyVectorStore(settings.vector_db_path)
    vector_store.initialize()

    if settings.app_env == 'testing':
        llm_client = TestLLMClient()
    else:
        llm_client = OpenAICompatibleLLMClient(
            settings.hf_token or '', settings.llm_model,
            settings.llm_timeout_seconds, settings.llm_max_output_tokens,
            settings.retry_attempts, settings.retry_backoff_seconds,
        )

    search_service = SearchService(vector_store, settings.min_similarity_score, settings.max_context_chars)
    evaluation_service = EvaluationService(settings.min_similarity_score)
    rag_service = RAGService(
        embedding_service, search_service, llm_client, settings.max_top_k, settings.default_top_k,
        settings.include_source_previews, settings.source_preview_chars,
    )
    document_service = DocumentService(
        vector_store, embedding_service, settings.max_upload_bytes, settings.chunk_size_chars, settings.chunk_overlap_chars,
    )
    history_service = HistoryService(settings.session_storage_path)
    history_service.initialize()
    session_manager = SessionManager(settings.session_storage_path, settings.max_session_documents)
    session_manager.initialize()
    chat_service = ChatService(
        embedding_service, search_service, llm_client, history_service, session_manager,
        settings.max_top_k, settings.default_top_k, settings.include_source_previews,
        settings.source_preview_chars, settings.chat_history_max_messages,
        settings.chat_history_max_chars, settings.recent_messages_count, settings.summary_threshold,
    )
    return embedding_service, vector_store, llm_client, rag_service, document_service, evaluation_service, history_service, session_manager, chat_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        components = build_components()
        (app.state.embedding_service, app.state.vector_store, app.state.llm_client, app.state.rag_service, app.state.document_service, app.state.evaluation_service, app.state.history_service, app.state.session_manager, app.state.chat_service) = components
        app.state.rate_limiter = InMemoryRateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)
        logger.info('application_started')
    except Exception as e:
        logger.exception('application_startup_failed')
        raise RuntimeError(f"Failed to start application: {e}")

    yield

    # Cleanup on shutdown
    for client in (app.state.embedding_service.provider, app.state.llm_client):
        close = getattr(client, 'close', None)
        if close:
            try:
                close()
            except Exception:
                pass
    logger.info('application_stopped')


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description='Production-oriented, document-grounded RAG question answering API.',
    lifespan=lifespan,
    docs_url='/docs',
    redoc_url='/redoc',
    openapi_url='/openapi.json',
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'DELETE', 'OPTIONS'],
    allow_headers=['*'],
    expose_headers=['X-Request-ID'],
    max_age=600
)


@app.middleware('http')
async def request_context_middleware(request: Request, call_next):
    supplied = request.headers.get(settings.request_id_header)
    try:
        request_id = str(UUID(supplied)) if supplied else new_request_id()
    except (ValueError, AttributeError):
        request_id = new_request_id()
    token = request_id_var.set(request_id)
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        if settings.api_key_enabled:
            provided = request.headers.get('X-API-Key')
            if not provided or provided != settings.api_key:
                return JSONResponse(status_code=401, content={'error': {'code': 'UNAUTHORIZED', 'message': 'API authentication required.', 'request_id': request_id}})
        if settings.rate_limit_enabled and request.url.path not in {'/health', '/ready'}:
            key = request.client.host if request.client else 'unknown'
            if not request.app.state.rate_limiter.allow(key):
                return JSONResponse(status_code=429, content={'error': {'code': 'RATE_LIMITED', 'message': 'Rate limit exceeded.', 'request_id': request_id}})
        response = await call_next(request)
        response.headers[settings.request_id_header] = request_id
        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info('request_complete', extra={'endpoint': request.url.path, 'duration_ms': duration_ms})
        return response
    finally:
        request_id_var.reset(token)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    request_id = getattr(request.state, 'request_id', new_request_id())
    logger.error('application_error', extra={'error_code': exc.code, 'endpoint': request.url.path})
    return JSONResponse(status_code=exc.status_code, content={'error': {'code': exc.code, 'message': exc.message, 'request_id': request_id}})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, 'request_id', new_request_id())
    messages = []
    for error in exc.errors():
        loc = '.'.join(str(part) for part in error.get('loc', []))
        messages.append(f'{loc}: {error.get("msg", "invalid value")}')
    return JSONResponse(status_code=422, content={'error': {'code': 'INVALID_REQUEST', 'message': '; '.join(messages), 'request_id': request_id}})


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, 'request_id', new_request_id())
    logger.exception('unhandled_exception', extra={'endpoint': request.url.path})
    return JSONResponse(status_code=500, content={'error': {'code': 'INTERNAL_ERROR', 'message': 'An unexpected internal error occurred.', 'request_id': request_id}})


app.include_router(health_router)
app.include_router(ask_router)
app.include_router(documents_router)
app.include_router(chat_router)

# Serve frontend from ../frontend
frontend_dir = Path(__file__).resolve().parent.parent.parent / 'frontend'
if frontend_dir.exists():
    app.mount('/static', StaticFiles(directory=str(frontend_dir)), name='static')
    @app.get('/', include_in_schema=False)
    async def serve_index():
        return FileResponse(str(frontend_dir / 'index.html'))
    @app.get('/script.js', include_in_schema=False)
    async def serve_script():
        return FileResponse(str(frontend_dir / 'script.js'))
    @app.get('/style.css', include_in_schema=False)
    async def serve_style():
        return FileResponse(str(frontend_dir / 'style.css'))


@app.get('/', tags=['system'], summary='API information')
async def root():
    return {
        'name': settings.app_name,
        'version': settings.app_version,
        'docs': '/docs',
        'redoc': '/redoc',
        'health': '/health',
        'ready': '/ready',
        'ask': '/ask',
        'evaluate': '/evaluate',
        'ask_evaluate': '/ask-evaluate',
        'chat': '/chat',
        'attach_document': '/attach-document',
        'history': '/history/{session_id}',
        'clear_history': '/history/{session_id}',
        'summarize_history': '/summarize-history',
        'openapi': '/openapi.json',
    }
