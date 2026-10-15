from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.schemas.response import HealthResponse, ReadyResponse

settings = get_settings()
router = APIRouter(tags=['system'])


@router.get('/health', response_model=HealthResponse, summary='Liveness check')
async def health():
    return HealthResponse(status='ok', service=settings.app_name, version=settings.app_version)


@router.get('/ready', response_model=ReadyResponse, responses={503: {'model': ReadyResponse}}, summary='Dependency readiness check')
async def ready(request: Request):
    components = {
        'vector_store': request.app.state.vector_store.health(),
        'embedding_service': request.app.state.embedding_service.health(),
        'llm': request.app.state.llm_client.health(),
    }
    ready_status = all(value == 'ready' for value in components.values())
    body = ReadyResponse(status='ready' if ready_status else 'not_ready', service=settings.app_name, version=settings.app_version, components=components)
    return body if ready_status else JSONResponse(status_code=503, content=body.model_dump())
