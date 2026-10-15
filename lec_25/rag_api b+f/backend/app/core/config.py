from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', case_sensitive=False, extra='ignore')

    app_name: str = 'RAG Question Answering API'
    app_version: str = '1.0.0'
    app_env: Literal['development', 'testing', 'staging', 'production'] = 'development'
    host: str = '0.0.0.0'
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = 'INFO'

    llm_model: str = 'openai/gpt-oss-20b'
    hf_token: str | None = None
    llm_timeout_seconds: float = Field(default=15.0, gt=0, le=300)
    llm_max_output_tokens: int = Field(default=700, ge=1, le=8192)

    embedding_provider: Literal['hash', 'openai_compatible'] = 'hash'
    embedding_model: str = 'text-embedding-3-small'
    embedding_api_key: str | None = None
    embedding_base_url: str = 'https://api.openai.com/v1'
    embedding_dimension: int = Field(default=384, ge=8, le=8192)
    embedding_timeout_seconds: float = Field(default=10.0, gt=0, le=300)

    vector_db_type: Literal['sqlite_numpy'] = 'sqlite_numpy'
    vector_db_path: str = './vector_db/vectors.sqlite3'

    default_top_k: int = Field(default=3, ge=1, le=50)
    max_top_k: int = Field(default=10, ge=1, le=100)
    min_similarity_score: float = Field(default=0.05, ge=-1.0, le=1.0)
    max_context_chars: int = Field(default=12000, ge=1000, le=200000)
    max_question_length: int = Field(default=4000, ge=10, le=50000)
    source_preview_chars: int = Field(default=300, ge=0, le=2000)
    include_source_previews: bool = False
    debug_mode: bool = False

    request_id_header: str = 'X-Request-ID'
    cors_origins: str = 'http://localhost:3000,http://localhost:5173,http://localhost:8000,http://127.0.0.1:8000,file://'
    api_key_enabled: bool = False
    api_key: str | None = None
    rate_limit_enabled: bool = False
    rate_limit_requests: int = Field(default=60, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)

    max_upload_bytes: int = Field(default=10_000_000, ge=1024)
    chunk_size_chars: int = Field(default=800, ge=200, le=10000)
    chunk_overlap_chars: int = Field(default=150, ge=0, le=1000)

    retry_attempts: int = Field(default=2, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=0.25, ge=0, le=10)

    # Conversational RAG
    chat_history_max_messages: int = Field(default=20, ge=2, le=200)
    chat_history_max_chars: int = Field(default=16000, ge=1000, le=200000)
    recent_messages_count: int = Field(default=8, ge=1, le=50)
    summary_threshold: int = Field(default=24, ge=4, le=500)
    max_session_documents: int = Field(default=20, ge=1, le=100)
    session_storage_path: str = './vector_db/sessions.sqlite3'
    max_file_path_length: int = Field(default=1000, ge=100, le=4096)
    allowed_file_extensions: str = '.txt,.md,.markdown,.pdf,.docx'

    @field_validator('chunk_overlap_chars')
    @classmethod
    def overlap_smaller_than_chunk(cls, value: int, info):
        chunk_size = info.data.get('chunk_size_chars', 1200)
        if value >= chunk_size:
            raise ValueError('chunk_overlap_chars must be smaller than chunk_size_chars')
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [x.strip() for x in self.cors_origins.split(',') if x.strip()]


    def validate_runtime(self) -> None:
        if self.app_env == 'production' and self.debug_mode:
            raise ValueError('debug_mode must be false in production')
        if self.app_env == 'production' and self.api_key_enabled and not self.api_key:
            raise ValueError('API_KEY is required when API_KEY_ENABLED=true')
        if self.app_env != 'testing' and not self.hf_token:
            raise ValueError('HF_TOKEN is required for Hugging Face inference.')
        if self.embedding_provider == 'openai_compatible' and not self.embedding_api_key:
            raise ValueError('EMBEDDING_API_KEY is required for openai_compatible embedding provider')
        if self.max_top_k < self.default_top_k:
            raise ValueError('MAX_TOP_K must be >= DEFAULT_TOP_K')


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
