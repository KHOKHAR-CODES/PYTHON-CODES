from app.core.exceptions import EmbeddingError, LLMError, LLMTimeoutError, VectorStoreUnavailableError


def test_embedding_failure_is_clean(client, monkeypatch):
    def fail(_question):
        raise EmbeddingError('test embedding failure')
    monkeypatch.setattr(client.app.state.embedding_service, 'embed_query', fail)
    response = client.post('/ask', json={'question': 'hello'})
    assert response.status_code == 502
    assert response.json()['error']['code'] == 'EMBEDDING_ERROR'
    assert 'test embedding failure' in response.json()['error']['message']


def test_vector_store_failure_is_clean(client, monkeypatch):
    def fail(*_args):
        raise VectorStoreUnavailableError('test vector failure')
    monkeypatch.setattr(client.app.state.vector_store, 'search', fail)
    response = client.post('/ask', json={'question': 'hello'})
    assert response.status_code == 503
    assert response.json()['error']['code'] == 'VECTOR_STORE_UNAVAILABLE'


def test_llm_failure_is_clean(client, monkeypatch):
    def fail(*_args):
        raise LLMError('test llm failure')
    monkeypatch.setattr(client.app.state.llm_client, 'generate_answer', fail)
    content = open('data/rag.txt', encoding='utf-8').read()
    client.post('/documents/index', json={'document_name': 'sample.txt', 'content': content})
    response = client.post('/ask', json={'question': 'What is FastAPI?', 'top_k': 1})
    assert response.status_code == 502
    assert response.json()['error']['code'] == 'LLM_ERROR'


def test_llm_timeout_is_clean(client, monkeypatch):
    def fail(*_args):
        raise LLMTimeoutError()
    monkeypatch.setattr(client.app.state.llm_client, 'generate_answer', fail)
    content = open('data/rag.txt', encoding='utf-8').read()
    client.post('/documents/index', json={'document_name': 'sample.txt', 'content': content})
    response = client.post('/ask', json={'question': 'What is FastAPI?', 'top_k': 1})
    assert response.status_code == 504
    assert response.json()['error']['code'] == 'LLM_TIMEOUT'
