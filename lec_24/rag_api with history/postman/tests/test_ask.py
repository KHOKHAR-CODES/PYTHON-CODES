def index_sample(client):
    content = open('data/sample.txt', encoding='utf-8').read()
    response = client.post('/documents/index', json={'document_name': 'sample.txt', 'content': content})
    assert response.status_code == 200, response.text


def test_valid_ask(client):
    index_sample(client)
    response = client.post('/ask', json={'question': 'What is FastAPI?', 'top_k': 3})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['request_id']
    assert body['answer']
    assert isinstance(body['sources'], list)
    assert body['metadata']['selected_chunks'] >= 1


def test_rag_question(client):
    index_sample(client)
    response = client.post('/ask', json={'question': 'What is Retrieval-Augmented Generation?', 'top_k': 3})
    assert response.status_code == 200
    assert 'retrieved' in response.json()['answer'].lower()


def test_irrelevant_question_is_controlled(client):
    index_sample(client)
    response = client.post('/ask', json={'question': 'What is the weather today?', 'top_k': 3, 'similarity_threshold': 0.95})
    assert response.status_code == 200
    assert response.json()['sources'] == []
    assert 'not contain enough information' in response.json()['answer'].lower()


def test_include_sources_can_be_disabled(client):
    index_sample(client)
    response = client.post('/ask', json={'question': 'What is FastAPI?', 'top_k': 1, 'include_sources': False})
    assert response.status_code == 200
    assert response.json()['sources'] == []
