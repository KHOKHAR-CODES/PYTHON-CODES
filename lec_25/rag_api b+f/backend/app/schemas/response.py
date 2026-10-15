from typing import Any
from pydantic import BaseModel, Field


class Source(BaseModel):
    document: str
    document_id: str
    chunk_id: str
    score: float
    page: int | None = None
    text_preview: str | None = None


class AskMetadata(BaseModel):
    retrieved_chunks: int
    selected_chunks: int
    model: str
    latency_ms: int
    embedding_latency_ms: int | None = None
    retrieval_latency_ms: int | None = None
    llm_latency_ms: int | None = None


class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    metadata: AskMetadata


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    service: str
    version: str
    components: dict[str, str]


class RootResponse(BaseModel):
    name: str
    version: str
    docs: str
    redoc: str
    health: str
    ready: str
    ask: str
    evaluate: str
    ask_evaluate: str
    openapi: str


class DocumentInfo(BaseModel):
    document_id: str
    document_name: str
    content_hash: str
    chunks: int
    created_at: str
    source: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted: bool


class DocumentUploadResponse(BaseModel):
    document_id: str
    document_name: str
    chunks_indexed: int
    content_hash: str


class EvaluationResponse(BaseModel):
    relevance_score: float = Field(ge=0.0, le=1.0)
    source_available: bool
    answer_supported: bool
    status: str


class AskEvaluateResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    retrieved_chunks: int
    evaluation: EvaluationResponse


class ChatMetadata(AskMetadata):
    history_messages: int
    session_documents: int
    summary_used: bool


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    question: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    metadata: ChatMetadata


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str


class HistoryResponse(BaseModel):
    session_id: str
    summary: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)


class SessionDocumentResponse(BaseModel):
    document_id: str
    document_name: str
    chunks_indexed: int
    content_hash: str
    attached: bool


class SessionDocumentsResponse(BaseModel):
    session_id: str
    documents: list[DocumentInfo] = Field(default_factory=list)


class SummaryResponse(BaseModel):
    session_id: str
    summary: str
    summarized_messages: int
