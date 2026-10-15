def test_upload_list_get_delete_document(client):
    data = b'FastAPI is a Python API framework.'
    response = client.post('/documents/index', json={'document_name': 'notes.txt', 'content': data.decode()})
    assert response.status_code == 200
    body = response.json()
    assert body['chunks_indexed'] >= 1
    document_id = body['document_id']

    listed = client.get('/documents')
    assert listed.status_code == 200
    assert listed.json()['documents'][0]['document_id'] == document_id

    fetched = client.get(f'/documents/{document_id}')
    assert fetched.status_code == 200
    assert fetched.json()['document_id'] == document_id

    deleted = client.delete(f'/documents/{document_id}')
    assert deleted.status_code == 200
    assert deleted.json()['deleted'] is True


def test_duplicate_upload_does_not_duplicate_chunks(client):
    data = b'Unique content for duplicate prevention.'
    first = client.post('/documents/index', json={'document_name': 'a.txt', 'content': data.decode()}).json()
    second = client.post('/documents/index', json={'document_name': 'b.txt', 'content': data.decode()}).json()
    assert second['document_id'] == first['document_id']
    assert second['chunks_indexed'] == first['chunks_indexed']
