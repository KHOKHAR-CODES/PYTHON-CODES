from fastapi import APIRouter, Request, UploadFile, File, Form

from app.core.exceptions import InvalidRequestError
from app.core.config import get_settings
from app.schemas.request import DocumentIndexRequest
from app.schemas.response import DocumentDeleteResponse, DocumentInfo, DocumentListResponse, DocumentUploadResponse, ErrorResponse, SessionDocumentResponse

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
    request.app.state.session_manager.detach_deleted_document(document_id)
    return {'document_id': document_id, 'deleted': True}


@router.post('/upload-session', response_model=SessionDocumentResponse, responses={400: {'model': ErrorResponse}, 413: {'model': ErrorResponse}}, summary='Upload, index, and attach a document to a chat session')
async def upload_session_document(
    request: Request,
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Browser-friendly document upload. Saves the file inside the approved uploads directory."""
    import os
    import re
    from pathlib import Path

    settings = get_settings()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', session_id):
        raise InvalidRequestError('Invalid session_id.')
    if not file.filename:
        raise InvalidRequestError('A document file is required.')

    suffix = Path(file.filename).suffix.lower()
    allowed = {x.strip().lower() for x in settings.allowed_file_extensions.split(',') if x.strip()}
    if suffix not in allowed:
        raise InvalidRequestError(f'Unsupported file type: {suffix or "none"}.')

    upload_dir = Path('./uploads')
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', Path(file.filename).name).strip('._')
    if not safe_name:
        raise InvalidRequestError('Invalid file name.')

    destination = upload_dir / f'{session_id}_{safe_name}'
    total = 0
    try:
        with destination.open('wb') as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise InvalidRequestError(
                        f'Document exceeds maximum size of {settings.max_upload_bytes} bytes.'
                    )
                handle.write(chunk)

        request.app.state.history_service.ensure_session(session_id)
        result = request.app.state.document_service.index_file(
            str(destination),
            allowed_roots=['./data', './uploads'],
            allowed_extensions=allowed,
        )
        try:
            request.app.state.session_manager.attach_document(
                session_id,
                result['document_id'],
                __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            )
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc

        return SessionDocumentResponse(**result, attached=True)
    finally:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
