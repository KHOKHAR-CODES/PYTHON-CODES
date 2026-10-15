
def index_sample(client):
    content = open('data/rag.txt', encoding='utf-8').read()
    response = client.post('/documents/index', json={'document_name':'rag.txt','content':content})
    assert response.status_code == 200
    return response.json()['document_id']

def attach(client, session_id, path='data/rag.txt'):
    response = client.post('/attach-document', json={'session_id':session_id,'file_path':path})
    assert response.status_code == 200, response.text
    return response.json()

def test_chat_and_followup(client):
    attach(client, 'student_001')
    first = client.post('/chat', json={'session_id':'student_001','question':'What is RAG?','top_k':2})
    assert first.status_code == 200, first.text
    second = client.post('/chat', json={'session_id':'student_001','question':'How does it use embeddings?','top_k':2})
    assert second.status_code == 200, second.text
    history = client.get('/history/student_001').json()
    assert len(history['messages']) == 4
    assert history['messages'][0]['content'] == 'What is RAG?'
    assert 'embeddings' in second.json()['answer'].lower()

def test_session_isolation(client):
    attach(client, 'student_001')
    client.post('/chat', json={'session_id':'student_001','question':'What is RAG?'})
    other = client.get('/history/student_002')
    assert other.status_code == 200
    assert other.json()['messages'] == []

def test_document_isolation(client):
    attach(client, 'student_001')
    # student_002 has no attached document, so retrieval must not see student_001's vector.
    response = client.post('/chat', json={'session_id':'student_002','question':'What is RAG?','similarity_threshold':0.0})
    assert response.status_code == 200
    assert response.json()['sources'] == []
    assert 'not contain enough information' in response.json()['answer'].lower()

def test_clear_history_preserves_documents(client):
    attach(client, 'student_001')
    client.post('/chat', json={'session_id':'student_001','question':'What is RAG?'})
    cleared = client.delete('/history/student_001')
    assert cleared.status_code == 200
    assert cleared.json()['messages'] == []
    docs = client.get('/session/student_001/documents').json()['documents']
    assert len(docs) == 1
    followup = client.post('/chat', json={'session_id':'student_001','question':'What is RAG?'})
    assert followup.status_code == 200
    assert followup.json()['sources']

def test_attach_supported_file_formats(client):
    for path in ['data/rag.txt', 'data/rag.pdf', 'data/rag.docx']:
        response = client.post('/attach-document', json={'session_id':'student_formats','file_path':path})
        assert response.status_code == 200, (path, response.text)

def test_path_traversal_rejected(client):
    response = client.post('/attach-document', json={'session_id':'student_001','file_path':'../.env'})
    assert response.status_code == 400

def test_invalid_session_rejected(client):
    response = client.post('/chat', json={'session_id':'bad session','question':'hello'})
    assert response.status_code == 422

def test_summarize_history(client):
    attach(client, 'student_summary')
    client.post('/chat', json={'session_id':'student_summary','question':'What is RAG?'})
    response = client.post('/summarize-history', json={'session_id':'student_summary'})
    assert response.status_code == 200
    assert response.json()['summarized_messages'] == 2
    history = client.get('/history/student_summary').json()
    assert history['summary']
