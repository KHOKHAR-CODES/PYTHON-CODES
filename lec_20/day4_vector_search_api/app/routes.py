"""API route definitions for the vector search service."""

from fastapi import APIRouter, HTTPException

from app.schemas import (
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
)
from app.search_service import (
    SearchServiceError,
    index_file_pipeline,
    index_is_empty,
    search_pipeline,
)

router = APIRouter()


@router.post("/index-file", response_model=IndexResponse)
def index_file(request: IndexRequest) -> IndexResponse:
    """Index a file: load, clean, split, embed, and store its content.

    Args:
        request: Contains the file_path to index.

    Returns:
        An IndexResponse summarizing how many chunks were created and stored.

    Raises:
        HTTPException: 400 if the file is missing, unsupported, or fails to index.
    """
    try:
        result = index_file_pipeline(request.file_path)
    except SearchServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IndexResponse(
        file_name=result["file_name"],
        total_chunks=result["total_chunks"],
        vectors_stored=result["vectors_stored"],
        status="success",
    )


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Search the indexed content for chunks relevant to a question.

    Args:
        request: Contains the question and top_k number of results to return.

    Returns:
        A SearchResponse with the question and matching results.

    Raises:
        HTTPException: 400 if top_k is invalid or the index is empty.
    """
    if request.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be a positive integer.")

    if index_is_empty():
        raise HTTPException(
            status_code=400,
            detail="The vector index is empty. Index at least one file before searching.",
        )

    try:
        results = search_pipeline(request.question, request.top_k)
    except SearchServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SearchResponse(question=request.question, results=results)
