from dataclasses import dataclass


@dataclass
class AppError(Exception):
    code: str
    message: str
    status_code: int = 500

    def __str__(self) -> str:
        return self.message


class InvalidRequestError(AppError):
    def __init__(self, message: str):
        super().__init__('INVALID_REQUEST', message, 400)


class ConfigurationError(AppError):
    def __init__(self, message: str):
        super().__init__('CONFIGURATION_ERROR', message, 503)


class VectorStoreUnavailableError(AppError):
    def __init__(self, message: str = 'Vector store is unavailable.'):
        super().__init__('VECTOR_STORE_UNAVAILABLE', message, 503)


class EmbeddingError(AppError):
    def __init__(self, message: str = 'Embedding service failed.'):
        super().__init__('EMBEDDING_ERROR', message, 502)


class LLMError(AppError):
    def __init__(self, message: str = 'Language model provider failed.'):
        super().__init__('LLM_ERROR', message, 502)


class LLMTimeoutError(AppError):
    def __init__(self, message: str = 'Language model request timed out.'):
        super().__init__('LLM_TIMEOUT', message, 504)


class NoRelevantContextError(AppError):
    def __init__(self, message: str = 'The available documents do not contain enough information to answer this question.'):
        super().__init__('NO_RELEVANT_CONTEXT', message, 200)


class MalformedLLMResponseError(LLMError):
    def __init__(self, message: str = 'Language model returned an invalid response.'):
        super().__init__(message)
        self.code = 'LLM_ERROR'
