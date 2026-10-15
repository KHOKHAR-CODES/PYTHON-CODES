from pydantic import BaseModel, ConfigDict, Field, field_validator


class AskRequest(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            'examples': [
                {'question': 'What is FastAPI?', 'top_k': 3},
                {'question': 'What is Retrieval-Augmented Generation?', 'top_k': 3, 'similarity_threshold': 0.35},
            ]
        },
    )

    question: str = Field(..., min_length=1, max_length=4000, description='Question to answer from indexed documents.')
    top_k: int = Field(default=3, ge=1, le=100, description='Maximum number of candidate chunks to retrieve.')
    similarity_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    include_sources: bool = True

    @field_validator('question')
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = ' '.join(value.split())
        if not value:
            raise ValueError('question must not be empty')
        return value


class DocumentIndexRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    document_name: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1, max_length=10_000_000)
    source: str | None = Field(default=None, max_length=1000)
    page: int | None = Field(default=None, ge=1)

    @field_validator('document_name', 'content')
    @classmethod
    def trim_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('value must not be empty')
        return value



class EvaluationRequest(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={'examples': [{'question': 'What is FastAPI?', 'answer': 'FastAPI is a Python framework for building APIs.', 'sources': ['sample.pdf']}]},
    )

    question: str = Field(..., min_length=1, max_length=4000)
    answer: str = Field(..., min_length=1, max_length=10000)
    sources: list[str] = Field(default_factory=list)

    @field_validator('question', 'answer')
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('value must not be empty')
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', json_schema_extra={'examples': [
        {'session_id': 'student_001', 'question': 'What is RAG?'},
        {'session_id': 'student_001', 'question': 'How does it use embeddings?'},
    ]})
    session_id: str = Field(..., min_length=1, max_length=128, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$')
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=100)
    similarity_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    include_sources: bool = True

    @field_validator('question')
    @classmethod
    def normalize_chat_question(cls, value: str) -> str:
        value = ' '.join(value.split())
        if not value:
            raise ValueError('question must not be empty')
        return value


class AttachDocumentRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', json_schema_extra={'examples': [
        {'session_id': 'student_001', 'file_path': 'data/rag.pdf'}
    ]})
    session_id: str = Field(..., min_length=1, max_length=128, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$')
    file_path: str = Field(..., min_length=1, max_length=1000)

    @field_validator('file_path')
    @classmethod
    def normalize_path(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('file_path must not be empty')
        return value


class SummarizeHistoryRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session_id: str = Field(..., min_length=1, max_length=128, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$')
