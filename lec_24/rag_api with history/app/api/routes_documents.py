from fastapi import APIRouter, Request

from app.core.exceptions import InvalidRequestError
from app.schemas.request import DocumentIndexRequest
from app.schemas.response import DocumentDeleteResponse, DocumentInfo, DocumentListResponse, DocumentUploadResponse, ErrorResponse

router = APIRouter(prefix='/documents', tags=['documents'])


@router.post('/index', response_model=DocumentUploadResponse, responses={400: {'model': ErrorResponse}}, summary='Index text or Markdown content')
def index_document(request: Request, payload: DocumentIndexRequest):
    return request.app.state.document_service.index_text(payload.document_name, payload.content, payload.source, payload.page)


@router.get('', response_model=DocumentListResponse, summary='List indexed documents')
def list_documents(request: Request):
    return {'documents': request.app.state.vector_store.list_documents()}


@router.get('/{document_id}', response_model=DocumentInfo, responses={400: {'model': ErrorResponse}}, summary='Get indexed document metadata')
def get_document(request: Request, document_id: str):
    documents = request.app.state.vector_store.list_documents()
    match = next((d for d in documents if d['document_id'] == document_id), None)
    if not match:
        raise InvalidRequestError('Document not found.')
    return match


@router.delete('/{document_id}', response_model=DocumentDeleteResponse, responses={400: {'model': ErrorResponse}}, summary='Delete a document and its chunks')
def delete_document(request: Request, document_id: str):
    deleted = request.app.state.vector_store.delete_document(document_id)
    if not deleted:
        raise InvalidRequestError('Document not found.')
    return {'document_id': document_id, 'deleted': True}
