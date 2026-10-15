def test_empty_question(client):
    response = client.post('/ask', json={'question': '', 'top_k': 3})
    assert response.status_code == 422
    body = response.json()
    assert body['error']['code'] == 'INVALID_REQUEST'
    assert body['error']['request_id']


def test_whitespace_question(client):
    response = client.post('/ask', json={'question': '   ', 'top_k': 3})
    assert response.status_code == 422
    assert response.json()['error']['code'] == 'INVALID_REQUEST'


def test_top_k_zero(client):
    response = client.post('/ask', json={'question': 'hello', 'top_k': 0})
    assert response.status_code == 422


def test_top_k_over_api_schema_max(client):
    response = client.post('/ask', json={'question': 'hello', 'top_k': 101})
    assert response.status_code == 422


def test_unknown_fields_rejected(client):
    response = client.post('/ask', json={'question': 'hello', 'unexpected': True})
    assert response.status_code == 422


def test_request_id_header_is_preserved_when_uuid(client):
    request_id = '123e4567-e89b-12d3-a456-426614174000'
    response = client.post('/ask', headers={'X-Request-ID': request_id}, json={'question': 'hello', 'similarity_threshold': 0.99})
    assert response.status_code == 200
    assert response.json()['request_id'] == request_id
