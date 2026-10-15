def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ok'
    assert body['service']
    assert body['version']


def test_ready(client):
    response = client.get('/ready')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ready'
    assert body['components'] == {'vector_store': 'ready', 'embedding_service': 'ready', 'llm': 'ready'}


def test_root(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.json()['ask'] == '/ask'
